"""Combine the fact scores of several runs into one run.

Each --group is a list of <run folder>:<seed> members whose scores are averaged (a fact
missing from a member counts as 0 there). With two groups, the weight of the first one
is chosen on dev, from 0 to 1 by steps of 0.1; with one group there is nothing to choose.
The fact threshold is then chosen on dev. Examples:

  # three seeds of one model
  combine_runs.py --run doc-ens --group results/doc:0 results/doc:1 results/doc:2
  # a document model and PromptORE
  combine_runs.py --run doc+m2 --group results/doc:0 --group results/promptore-classifier:0

Test is combined only if every member has test scores, with the dev weight and threshold.
"""
import argparse
import json
from pathlib import Path

import torch

from kg_extraction.data.linked_docred import load_split
from kg_extraction.evaluation.relations import gold_facts
from kg_extraction.evaluation.runs import write_shared_run
from kg_extraction.relations.doc_model import best_threshold


def group_scores(members, split):
    """Return the mean score of each fact over the members, or None if a file is missing."""
    total = {}
    for member in members:
        run_dir, seed = member.rsplit(":", 1)
        path = Path(run_dir) / f"scores_{split}_seed{seed}.pt"
        if not path.exists():
            return None
        for fact, score in torch.load(path, weights_only=False).items():
            total[fact] = total.get(fact, 0.0) + score / len(members)
    return total


def weighted(groups, weights):
    total = {}
    for scores, weight in zip(groups, weights):
        if weight:
            for fact, score in scores.items():
                total[fact] = total.get(fact, 0.0) + weight * score
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--group", nargs="+", action="append", required=True)
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--runs-dir", default="results")
    parser.add_argument("--author", default="ZHU-Laurent")
    parser.add_argument("--seed", type=int, default=0, help="seed number of the output files")
    args = parser.parse_args()

    dev = load_split(args.data_dir, "dev")
    gold = gold_facts(dev)
    groups = [group_scores(members, "dev") for members in args.group]
    candidates = [[1.0]] if len(groups) == 1 else [[1 / len(groups)] * len(groups)]
    if len(groups) == 2:
        candidates = [[step / 10, 1 - step / 10] for step in range(11)]
    search = []
    for weights in candidates:
        threshold, f1 = best_threshold(weighted(groups, weights), gold)
        search.append({"weights": weights, "threshold": round(threshold, 4), "dev_f1": f1})
        print(json.dumps(search[-1]), flush=True)
    best = max(search, key=lambda row: row["dev_f1"])

    docs, scores = {"dev": dev}, {"dev": weighted(groups, best["weights"])}
    test_groups = [group_scores(members, "test") for members in args.group]
    if all(group is not None for group in test_groups):
        docs["test"] = load_split(args.data_dir, "test")
        scores["test"] = weighted(test_groups, best["weights"])
    run_dir = Path(args.runs_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    for split, facts in scores.items():
        torch.save(facts, run_dir / f"scores_{split}_seed{args.seed}.pt")
    cards = [json.loads((Path(members[0].rsplit(":", 1)[0]) / "card.json").read_text())
             for members in args.group]
    card = {"run": args.run, "author": args.author,
            "method": "Combination of " + " and ".join(card["run"] for card in cards),
            "supervision": "supervised", "training_data": "Linked-Re-DocRED train labels",
            "models": "; ".join(dict.fromkeys(card["models"] for card in cards)),
            "description": f"Mean of the fact scores of {[len(m) for m in args.group]} members "
                           f"per group, group weights {best['weights']}; weights and fact "
                           "threshold chosen on dev"}
    kept = {split: {fact: score for fact, score in facts.items() if score >= best["threshold"]}
            for split, facts in scores.items()}
    write_shared_run(run_dir, docs, kept, card, args.seed)
    (run_dir / f"combine_report_seed{args.seed}.json").write_text(
        json.dumps({"groups": args.group, "best": best, "search": search}, indent=1))
    print(json.dumps({"best": best}))


if __name__ == "__main__":
    main()
