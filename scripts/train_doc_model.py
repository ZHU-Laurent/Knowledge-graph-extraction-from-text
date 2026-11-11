"""Train the document-level relation model (relations/doc_model.py) on frozen token features.

Each option adds one part to the ATLOP head, so that its gain can be measured alone:
--encoder (another frozen encoder), --token-layers, --pair-features, --pair-layers, --loss.
--tune-layers also trains the last layers of the encoder (BERT-like encoders only), from
the cached states that enter them.

The epoch and the fact threshold are chosen on dev. The run folder gets the facts for
the shared evaluator (scripts/evaluate.py) and the scores of every fact above 0.02
(scores_<split>_seed<N>.pt), which tune_thresholds.py and combine_runs.py read.
Test is predicted only with --test, with the epoch and threshold chosen on dev.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from kg_extraction.data.linked_docred import load_relation_names, load_split
from kg_extraction.evaluation.relations import gold_facts
from kg_extraction.evaluation.runs import write_shared_run
from kg_extraction.relations.doc_encoder import (
    MODEL_NAME, cached_features, default_device, pretrained_top_layers)
from kg_extraction.relations.doc_model import (
    DocRelationModel, adaptive_focal_loss, best_threshold, predict_scores, prepare_document,
    to_device)
from kg_extraction.relations.pair_classifier import adaptive_threshold_loss

LOSSES = {"atl": adaptive_threshold_loss, "afl": adaptive_focal_loss}


def features_path(directory, encoder, split, tune_layers=0):
    """bert-base-cased keeps its first cache path; other encoders get a sub-folder."""
    directory = Path(directory)
    if encoder != MODEL_NAME:
        directory = directory / encoder.replace("/", "--")
    return directory / (f"lower{tune_layers}_{split}.pt" if tune_layers else f"{split}.pt")


def load_items(args, split, docs, entity_types, device, relations=None):
    """Return the prepared documents of a split, kept on the device to avoid a copy per step."""
    features = cached_features(
        features_path(args.features_dir, args.encoder, split, args.tune_layers), docs,
        args.encoder, args.tune_layers)
    return [to_device(prepare_document(doc, doc_features, entity_types, relations), device)
            for doc, doc_features in zip(docs, features)]


def f1_at(facts, gold, threshold):
    kept = {fact for fact, score in facts.items() if score >= threshold}
    return round(200 * len(kept & gold) / max(len(kept) + len(gold), 1), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run folder")
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--models-dir", default="data/processed/models")
    parser.add_argument("--runs-dir", default="results")
    parser.add_argument("--author", default="ZHU-Laurent")
    parser.add_argument("--encoder", default=MODEL_NAME)
    parser.add_argument("--token-layers", type=int, default=0)
    parser.add_argument("--pair-layers", type=int, default=0)
    parser.add_argument("--pair-features", action="store_true")
    parser.add_argument("--loss", choices=sorted(LOSSES), default="atl")
    parser.add_argument("--tune-layers", type=int, default=0,
                        help="number of last encoder layers to train")
    parser.add_argument("--encoder-learning-rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4, help="documents per step")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="also predict test")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = default_device()
    relations = sorted(load_relation_names(args.data_dir))
    train = load_split(args.data_dir, "train")
    dev = load_split(args.data_dir, "dev")
    entity_types = sorted({entity["type"] for doc in train for entity in doc["entities"]})
    train_items = load_items(args, "train", train, entity_types, device, relations)
    dev_items = load_items(args, "dev", dev, entity_types, device)
    dev_gold = gold_facts(dev)

    top_layers = (pretrained_top_layers(args.encoder, args.tune_layers)
                  if args.tune_layers else None)
    hidden = train_items[0]["lower" if args.tune_layers else "states"][0].shape[-1]
    model = DocRelationModel(len(relations), hidden, len(entity_types), args.token_layers,
                             args.pair_layers, args.pair_features,
                             top_layers=top_layers).to(device)
    loss_function = LOSSES[args.loss]
    encoder_parameters = list(model.encoder.parameters()) if args.tune_layers else []
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    groups = [{"params": [parameter for parameter in model.parameters()
                          if id(parameter) not in encoder_ids]}]
    rates = [args.learning_rate]
    if encoder_parameters:
        groups.append({"params": encoder_parameters})
        rates.append(args.encoder_learning_rate)
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate)
    steps = args.epochs * -(-len(train_items) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, rates, steps, pct_start=0.06)
    model_path = Path(args.models_dir) / f"{args.run}_seed{args.seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.runs_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    history, best, best_facts, waited = [], None, None, 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = list(range(len(train_items)))
        random.shuffle(order)
        total, n_pairs = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = [train_items[index] for index in order[start:start + args.batch_size]]
            logits = torch.cat([model(item) for item in batch])
            labels = torch.cat([item["labels"] for item in batch]).float()
            loss = loss_function(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(labels)
            n_pairs += len(labels)

        facts = predict_scores(model, dev_items, dev, relations)
        threshold, f1 = best_threshold(facts, dev_gold)
        history.append({"epoch": epoch, "loss": round(total / n_pairs, 4),
                        "dev_f1_learned_threshold": f1_at(facts, dev_gold, 0.5),
                        "dev_f1": f1, "threshold": round(threshold, 4),
                        "seconds": round(time.time() - started)})
        print(json.dumps(history[-1]), flush=True)
        if best is None or f1 > best["dev_f1"]:
            best, best_facts, waited = history[-1], facts, 0
            torch.save({"state": model.state_dict(), "settings": model.settings,
                        "relations": relations, "entity_types": entity_types,
                        "encoder": args.encoder, "threshold": threshold}, model_path)
        else:
            waited += 1
            if waited >= args.patience:
                break

    report = {"settings": vars(args), "best": best, "history": history}
    (run_dir / f"train_report_seed{args.seed}.json").write_text(json.dumps(report, indent=2),
                                                                encoding="utf-8")
    scores = {"dev": best_facts}
    docs = {"dev": dev}
    if args.test:
        model.load_state_dict(torch.load(model_path, weights_only=False)["state"])
        docs["test"] = load_split(args.data_dir, "test")
        scores["test"] = predict_scores(
            model, load_items(args, "test", docs["test"], entity_types, device), docs["test"], relations)
    for split, facts in scores.items():
        torch.save(facts, run_dir / f"scores_{split}_seed{args.seed}.pt")
    parts = [f"last {args.tune_layers} encoder layers trained" if args.tune_layers else "",
             f"{args.token_layers} token layers" if args.token_layers else "",
             "pair features" if args.pair_features else "",
             f"{args.pair_layers} pair reasoning layers" if args.pair_layers else "",
             "adaptive focal loss" if args.loss == "afl" else ""]
    card = {"run": args.run, "author": args.author,
            "method": "Document relation model on "
                      + ("a partly fine-tuned encoder" if args.tune_layers else "frozen features"),
            "supervision": "supervised", "training_data": "Linked-Re-DocRED train labels",
            "models": f"{args.encoder} ({'lower layers frozen' if args.tune_layers else 'frozen'})",
            "description": "ATLOP head" + "".join(f" + {part}" for part in parts if part)
                           + "; epoch and fact threshold chosen on dev"}
    kept = {split: {fact: score for fact, score in facts.items() if score >= best["threshold"]}
            for split, facts in scores.items()}
    write_shared_run(run_dir, docs, kept, card, args.seed)
    print(json.dumps({"best": best}))


if __name__ == "__main__":
    main()
