"""Measure what a tuned fact threshold brings to a run of train_doc_model.py or combine_runs.py.

Three settings are compared on dev, from the scores saved in the run folder:
- learned: the threshold learned in training (score 0.5);
- global: one threshold for every relation;
- per relation: one threshold per relation, set by coordinate ascent on the micro F1.
  Relations with fewer than --min-gold dev facts keep the global threshold.

Tuning and scoring on the same documents is optimistic, mostly for the per-relation
thresholds (96 values). So each setting is also cross-fitted: the documents are split in
--folds folds, and each fold is scored with the thresholds tuned on the other folds.
--write saves the thresholds tuned on all of dev, for a later test run.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from kg_extraction.data.linked_docred import load_split
from kg_extraction.evaluation.relations import gold_facts


def f1(correct, predicted, gold):
    return 200 * correct / max(predicted + gold, 1)


def counts(facts, gold, thresholds, default):
    kept = {fact for fact, score in facts.items() if score >= thresholds.get(fact[3], default)}
    return len(kept & gold), len(kept)


def global_threshold(facts, gold):
    ranked = sorted(facts.items(), key=lambda fact: -fact[1])
    best, correct = (0.5, 0.0), 0
    for rank, (fact, score) in enumerate(ranked, start=1):
        correct += fact in gold
        if f1(correct, rank, len(gold)) > best[1]:
            best = (score, f1(correct, rank, len(gold)))
    return best[0]


def relation_thresholds(facts, gold, default, min_gold, passes=2):
    """Return {relation: threshold}, each chosen to maximize the micro F1 given the others."""
    by_relation = defaultdict(list)
    for fact, score in facts.items():
        by_relation[fact[3]].append((score, fact in gold))
    n_gold = defaultdict(int)
    for fact in gold:
        n_gold[fact[3]] += 1
    tuned = [relation for relation in by_relation if n_gold[relation] >= min_gold]
    curves = {}
    for relation in tuned:
        ranked = sorted(by_relation[relation], reverse=True)
        scores = np.array([score for score, _ in ranked])
        curves[relation] = (scores, np.cumsum([hit for _, hit in ranked]))
    thresholds = {}
    correct, predicted = counts(facts, gold, thresholds, default)
    for _ in range(passes):
        for relation in tuned:
            scores, hits = curves[relation]
            kept = int((scores >= thresholds.get(relation, default)).sum())
            other_correct = correct - (hits[kept - 1] if kept else 0)
            other_predicted = predicted - kept
            ranks = np.arange(1, len(scores) + 1)
            values = 2 * (other_correct + hits) / (other_predicted + ranks + len(gold))
            best = int(values.argmax())
            # Predicting nothing for the relation is also a candidate.
            if 2 * other_correct / max(other_predicted + len(gold), 1) >= values[best]:
                thresholds[relation], kept_now = 1.01, 0
            else:
                thresholds[relation], kept_now = float(scores[best]), best + 1
            correct = other_correct + (hits[kept_now - 1] if kept_now else 0)
            predicted = other_predicted + kept_now
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-gold", type=int, default=30)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    dev = load_split(args.data_dir, "dev")
    gold = gold_facts(dev)
    facts = torch.load(Path(args.run_dir) / f"scores_dev_seed{args.seed}.pt", weights_only=False)
    fold_of = np.random.default_rng(0).permutation(len(dev)) % args.folds

    def tune(fit_facts, fit_gold):
        default = global_threshold(fit_facts, fit_gold)
        return default, relation_thresholds(fit_facts, fit_gold, default, args.min_gold)

    default, thresholds = tune(facts, gold)
    totals = {name: np.zeros(2) for name in ("global", "per_relation")}
    for fold in range(args.folds):
        fit_facts = {fact: score for fact, score in facts.items() if fold_of[fact[0]] != fold}
        held_facts = {fact: score for fact, score in facts.items() if fold_of[fact[0]] == fold}
        held_gold = {fact for fact in gold if fold_of[fact[0]] == fold}
        fold_default, fold_thresholds = tune(
            fit_facts, {fact for fact in gold if fold_of[fact[0]] != fold})
        totals["global"] += counts(held_facts, held_gold, {}, fold_default)
        totals["per_relation"] += counts(held_facts, held_gold, fold_thresholds, fold_default)

    result = {
        "learned_threshold_f1": round(f1(*counts(facts, gold, {}, 0.5), len(gold)), 2),
        "global": {"threshold": round(default, 4),
                   "f1_same_documents": round(f1(*counts(facts, gold, {}, default), len(gold)), 2),
                   "f1_cross_fitted": round(f1(*totals["global"], len(gold)), 2)},
        "per_relation": {
            "tuned_relations": len(thresholds),
            "f1_same_documents": round(f1(*counts(facts, gold, thresholds, default), len(gold)), 2),
            "f1_cross_fitted": round(f1(*totals["per_relation"], len(gold)), 2)},
    }
    print(json.dumps(result, indent=1))
    if args.write:
        path = Path(args.run_dir) / f"thresholds_seed{args.seed}.json"
        path.write_text(json.dumps({"default": default, "relations": thresholds, **result},
                                   indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
