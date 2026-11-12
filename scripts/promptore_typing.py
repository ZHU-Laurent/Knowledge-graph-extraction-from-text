"""PromptORE as a relation typer, on the pairs detected by a document model.

The PromptORE methods of compare_methods.py lose most of their facts at detection:
their detector sees one prompt per pair. Here a run of train_doc_model.py decides which
pairs are related (pair score: its best relation score), and PromptORE only names the relation:

- classifier: P(relation | related) of the PromptORE classifiers (M2 typing);
- clusters: share of the relation in the k-means cluster of the pair (M1 typing).

Fact score: pair score x typing score. The fact threshold is chosen on dev. The script
also saves the fact scores of M2 itself, for scripts/combine_runs.py.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch

from compare_methods import CARDS, SPLITS, promptore_embeddings
from kg_extraction.data.linked_docred import load_relation_names, load_split
from kg_extraction.evaluation.relations import gold_facts
from kg_extraction.evaluation.runs import write_shared_run
from kg_extraction.pairs.candidates import allowed_type_pairs
from kg_extraction.relations.clustering import ClusterTyper
from kg_extraction.relations.probe import best_threshold, facts_above


def pair_scores(path, pairs):
    """Return, for each PromptORE pair, the best relation score the document model gave it."""
    best = defaultdict(float)
    for (index, head, tail, _), score in torch.load(path, weights_only=False).items():
        best[(index, head, tail)] = max(best[(index, head, tail)], score)
    return np.array([best.get(pair[:3], 0.0) for pair in pairs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("detector_run", help="run folder of train_doc_model.py")
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--embeddings-dir", default="data/processed/embeddings")
    parser.add_argument("--probe", default="data/processed/models/promptore_probe.joblib")
    parser.add_argument("--runs-dir", default="results")
    parser.add_argument("--author", default="ZHU-Laurent")
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--k", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    docs = {split: load_split(args.data_dir, split) for split in ("train", *SPLITS)}
    relations = sorted(load_relation_names(args.data_dir))
    data, _ = promptore_embeddings(args, docs, allowed_type_pairs(docs["train"]))
    train_pairs, train_embeddings = data["train"]
    probe = joblib.load(args.probe)
    detector = Path(args.detector_run)
    splits = [split for split in SPLITS
              if (detector / f"scores_{split}_seed{args.seed}.pt").exists()]

    m2_dir = Path(args.runs_dir) / CARDS["promptore_classifier"]["run"]
    for split in SPLITS:
        torch.save(facts_above(data[split][0], probe.scores(data[split][1]), relations, 0.02),
                   m2_dir / f"scores_{split}_seed{args.seed}.pt")

    related = np.array([bool(pair[3]) for pair in train_pairs])
    typer = ClusterTyper(relations, args.k, args.seed).fit(
        train_embeddings[related], [pair[3] for pair in train_pairs if pair[3]])
    typing = {"classifier": lambda embeddings: probe.typing.predict_proba(embeddings),
              "clusters": typer.scores}
    dev_gold = gold_facts(docs["dev"])
    detected = {split: pair_scores(detector / f"scores_{split}_seed{args.seed}.pt",
                                   data[split][0]) for split in splits}
    for name, typing_scores in typing.items():
        scores = {split: detected[split][:, None] * typing_scores(data[split][1])
                  for split in splits}
        threshold = best_threshold(data["dev"][0], scores["dev"], relations, dev_gold)
        facts = {split: facts_above(data[split][0], scores[split], relations, threshold)
                 for split in splits}
        base = CARDS[f"promptore_{name}"]
        run = f"{base['run']}-on-{detector.name}"
        card = {**base, "run": run, "author": args.author, "supervision": "supervised",
                "method": base["method"].replace(" (M", f" typing, pairs from {detector.name} (M"),
                "description": f"Pairs detected by {detector.name}, relation named by PromptORE "
                               f"({name}); fact threshold {threshold:.2f} chosen on dev"}
        run_dir = Path(args.runs_dir) / run
        write_shared_run(run_dir, docs, facts, card, args.seed)
        for split in splits:
            torch.save(facts_above(data[split][0], scores[split], relations, 0.02),
                       run_dir / f"scores_{split}_seed{args.seed}.pt")
        print(json.dumps({"run": run, "threshold": threshold, "dev_facts": len(facts["dev"])}))


if __name__ == "__main__":
    main()
