"""Train the relation classifier on gold entities, with a frozen document encoder.

Documents are encoded once and cached, and the pooled vectors of every pair are
computed once. The classifier head is trained on all train pairs, shuffled across
documents, and the epoch with the best dev F1 is kept.
"""
import argparse
import json
from pathlib import Path

import torch

from kg_extraction.data.linked_docred import load_relation_names, load_split
from kg_extraction.evaluation.relations import relation_scores, train_fact_names
from kg_extraction.relations.doc_encoder import cached_features, default_device
from kg_extraction.relations.pair_classifier import (
    PairClassifier, adaptive_threshold_loss, corpus_inputs, pair_labels, predict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--output", default="data/processed/models/relation_model.pt")
    parser.add_argument("--report", default="results/known_entities/train_report.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024, help="pairs per step")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = default_device()
    relations = sorted(load_relation_names(args.data_dir))
    train = load_split(args.data_dir, "train")
    dev = load_split(args.data_dir, "dev")
    train_features = cached_features(Path(args.features_dir) / "train.pt", train)
    dev_features = cached_features(Path(args.features_dir) / "dev.pt", dev)
    train_names = train_fact_names(train)

    pairs, inputs = corpus_inputs(train_features, train)
    labels = torch.cat([pair_labels(doc, doc_pairs, relations).to(torch.uint8)
                        for doc, doc_pairs in zip(train, pairs)])
    dev_pairs, dev_inputs = corpus_inputs(dev_features, dev)
    del train_features, dev_features
    print(f"{len(inputs)} train pairs, {len(dev_inputs)} dev pairs", flush=True)
    model = PairClassifier(len(relations)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps = args.epochs * -(-len(inputs) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, steps,
                                                    pct_start=0.06)

    history, best, waited = [], None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(inputs))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            logits = model(inputs[batch].to(device))
            loss = adaptive_threshold_loss(logits, labels[batch].to(device).float())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(batch)

        scores = relation_scores(dev, predict(model, dev_pairs, dev_inputs, dev, relations),
                                 train_names)
        history.append({"epoch": epoch, "loss": round(total / len(inputs), 4), "dev": scores})
        print(json.dumps(history[-1]), flush=True)
        if best is None or scores["f1"] > best["dev"]["f1"]:
            best, waited = history[-1], 0
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "relations": relations}, args.output)
        else:
            waited += 1
            if waited >= args.patience:
                break

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": vars(args), "best": best, "history": history}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best["epoch"], "dev": best["dev"]}))


if __name__ == "__main__":
    main()
