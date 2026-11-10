"""Reference baseline and worked example of the prediction format for the main track.

No text is read. For each (head type, tail type, same sentence) key, train labels give P(relation | key);
a pair gets every relation whose probability passes a threshold tuned on dev_debug.
Each key is also written as a "cluster", so the clustering metrics have a floor to compare with.

    python scripts/baseline_typepair.py [--train-subset train] [--test]
    python scripts/evaluate.py results/baseline-typepair --leaderboard report/leaderboard.csv
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_extraction.data.redocred import RedocredGold  # noqa: E402
from kg_extraction.evaluation.metrics import fact_scores  # noqa: E402

BENCH = "data/processed/benchmark"


def read_pairs(subset):
    with open(os.path.join(BENCH, f"redocred_{subset}_pairs.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yield r["doc_id"], int(r["h"]), int(r["t"]), (r["h_type"], r["t_type"], r["same_sentence"])


def fit(train_subset):
    gold = RedocredGold(train_subset)
    n_key, n_key_rel = Counter(), defaultdict(Counter)
    for doc, h, t, key in read_pairs(train_subset):
        n_key[key] += 1
        for r in gold.pair_labels.get((doc, h, t), ()):
            n_key_rel[key][r] += 1
    return {k: {r: c / n_key[k] for r, c in rels.items()} for k, rels in n_key_rel.items()}


def predict(model, subset, threshold):
    facts, clusters = [], []
    for doc, h, t, key in read_pairs(subset):
        clusters.append((doc, h, t, "|".join(key)))
        facts.extend((doc, h, r, t) for r, p in model.get(key, {}).items() if p >= threshold)
    return facts, clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subset", default="train", choices=["train", "train_debug"])
    ap.add_argument("--test", action="store_true", help="also write test predictions (final evaluation only)")
    ap.add_argument("--out", default="results/baseline-typepair")
    args = ap.parse_args()

    model = fit(args.train_subset)
    dev_gold = RedocredGold("dev_debug")
    grid = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    scores = {th: fact_scores(dev_gold.facts, set(predict(model, "dev_debug", th)[0]))["micro_f1"] for th in grid}
    best = max(scores, key=scores.get)
    print("dev_debug micro F1 by threshold:", {k: round(v, 3) for k, v in scores.items()}, "-> best", best)

    os.makedirs(args.out, exist_ok=True)
    for subset in ["dev_debug", "dev"] + (["test"] if args.test else []):
        facts, clusters = predict(model, subset, best)
        with open(os.path.join(args.out, f"redocred_{subset}_facts.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "h", "r", "t"])
            w.writerows(facts)
        with open(os.path.join(args.out, f"redocred_{subset}_clusters.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "h", "t", "cluster"])
            w.writerows(clusters)
    with open(os.path.join(args.out, "card.json"), "w") as f:
        json.dump({"run": "baseline-typepair", "author": "shared", "method": "P(relation | entity types, same sentence)",
                   "supervision": "supervised", "training_data": f"Linked-Re-DocRED {args.train_subset} labels",
                   "models": "none (counting)",
                   "description": f"threshold {best} chosen on dev_debug; clusters = type-pair keys"}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
