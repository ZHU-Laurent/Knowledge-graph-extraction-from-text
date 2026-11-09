"""Score one run folder with the shared metrics and (optionally) add it to the leaderboard.

A run folder holds a card.json and prediction files named by convention (see docs/collaboration.md):

    results/<run>/card.json
    results/<run>/fewrel_<nyt|pubmed>_<k_true|k_est|other>_seed<N>.csv     columns: id,cluster
    results/<run>/redocred_<dev|dev_debug|test>_facts.csv                  columns: doc_id,h,r,t
    results/<run>/redocred_<dev|dev_debug|test>_clusters.csv  (optional)   columns: doc_id,h,t,cluster

Usage:
    python scripts/evaluate.py results/<run>
    python scripts/evaluate.py results/<run> --leaderboard report/leaderboard.csv
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_extraction.data.benchmark import read_fewrel_gold  # noqa: E402
from kg_extraction.data.redocred import RedocredGold, relation_buckets  # noqa: E402
from kg_extraction.evaluation.metrics import NO_RELATION, clustering_scores, fact_scores, prf  # noqa: E402

CARD_FIELDS = ["run", "author", "method", "supervision", "training_data", "models", "description"]
SUPERVISION = {"none", "labels_for_mapping", "supervised"}
BENCH = "data/processed/benchmark"
REDOCRED = "data/raw/Linked-Re-DocRED"


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_card(card):
    missing = [k for k in CARD_FIELDS if not card.get(k)]
    if missing:
        sys.exit(f"card.json is missing: {missing}")
    if card["supervision"] not in SUPERVISION:
        sys.exit(f"card.json supervision must be one of {sorted(SUPERVISION)}")


def score_fewrel(run_dir):
    groups = defaultdict(list)                      # (dataset, setting) -> [scores per seed]
    for path in sorted(glob.glob(os.path.join(run_dir, "fewrel_*_seed*.csv"))):
        m = re.match(r"fewrel_(nyt|pubmed)_(.+)_seed(\d+)\.csv$", os.path.basename(path))
        if not m:
            sys.exit(f"bad file name: {path}")
        dataset, setting, _ = m.groups()
        gold = read_fewrel_gold(os.path.join(BENCH, f"{dataset}_gold.csv"))
        pred = {r["id"]: r["cluster"] for r in read_csv(path)}
        if set(pred) != set(gold):
            sys.exit(f"{path}: ids differ from the benchmark "
                     f"({len(set(gold) - set(pred))} missing, {len(set(pred) - set(gold))} unknown)")
        ids = sorted(gold)
        groups[(dataset, setting)].append(clustering_scores([gold[i] for i in ids], [pred[i] for i in ids]))
    return groups


def score_redocred(run_dir):
    """Main track. Besides overall micro/macro scores it reports:
    - evid_*     facts whose gold annotation lists evidence sentences only (Re-DocRED added many facts
                 without evidence; predictions of those facts are then neither hits nor errors)
    - recall on pairs that share a sentence vs. pairs that do not
    - macro F1 on frequent / medium / rare relations (by train frequency)
    - detection P/R/F1: does the pair have at least one relation?
    - B3 / V / ARI on single-label related pairs, if a clusters file is given."""
    groups, buckets = {}, None
    for path in sorted(glob.glob(os.path.join(run_dir, "redocred_*_facts.csv"))):
        m = re.match(r"redocred_(dev|dev_debug|test)_facts\.csv$", os.path.basename(path))
        if not m:
            sys.exit(f"bad file name: {path} (allowed subsets: dev, dev_debug, test)")
        subset = m.group(1)
        gold = RedocredGold(subset, REDOCRED)
        buckets = buckets or relation_buckets(REDOCRED)
        pred = set()
        for row in read_csv(path):
            doc, h, r, t = row["doc_id"], int(row["h"]), row["r"], int(row["t"])
            if doc not in gold.docs:
                sys.exit(f"{path}: document {doc} is not in subset {subset}")
            if not (0 <= h < gold.n_entities(doc) and 0 <= t < gold.n_entities(doc)) or h == t:
                sys.exit(f"{path}: bad entity pair ({doc}, {h}, {t})")
            if r != NO_RELATION and r not in buckets:
                sys.exit(f"{path}: unknown relation {r}")
            pred.add((doc, h, r, t))

        s = fact_scores(gold.facts, pred)
        per_rel = s.pop("per_relation")
        ev = fact_scores(gold.facts, pred, ignore=gold.facts - gold.with_evidence)
        s.update({"evid_p": ev["micro_p"], "evid_r": ev["micro_r"], "evid_f1": ev["micro_f1"]})

        hits = {f for f in gold.facts if f in pred}
        for name, keep in (("intra", True), ("inter", False)):
            g = [f for f in gold.facts if gold.same_sentence(f[0], f[1], f[3]) == keep]
            s[f"recall_{name}_sentence"] = sum(f in hits for f in g) / len(g) if g else 0.0
        for b in ("frequent", "medium", "rare"):
            f1s = [v["f1"] for r, v in per_rel.items() if buckets.get(r) == b]
            s[f"macro_f1_{b}"] = float(np.mean(f1s)) if f1s else 0.0

        gold_pairs = set(gold.pair_labels)
        pred_pairs = {(d, h, t) for d, h, r, t in pred if r != NO_RELATION}
        s["detect_p"], s["detect_r"], s["detect_f1"] = prf(len(gold_pairs & pred_pairs), len(pred_pairs), len(gold_pairs))

        with open(os.path.join(run_dir, f"redocred_{subset}_per_relation.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relation", "bucket", "gold", "pred", "tp", "p", "r", "f1"])
            for r, v in sorted(per_rel.items(), key=lambda kv: -kv[1]["gold"]):
                w.writerow([r, buckets.get(r), v["gold"], v["pred"], v["tp"],
                            round(v["p"], 4), round(v["r"], 4), round(v["f1"], 4)])

        cl_path = os.path.join(run_dir, f"redocred_{subset}_clusters.csv")
        if os.path.exists(cl_path):
            clusters = {(r["doc_id"], int(r["h"]), int(r["t"])): r["cluster"] for r in read_csv(cl_path)}
            keys = [k for k, labels in gold.pair_labels.items() if len(labels) == 1 and k in clusters]
            if keys:
                cs = clustering_scores([next(iter(gold.pair_labels[k])) for k in keys], [clusters[k] for k in keys])
                s.update({f"cl_{k}": v for k, v in cs.items()})
        groups[(f"redocred_{subset}", "main")] = [s]
    return groups


def summarize(groups):
    rows = []
    for (dataset, setting), runs in sorted(groups.items()):
        row = {"dataset": dataset, "setting": setting, "n_seeds": len(runs)}
        for key in runs[0]:
            vals = np.array([r[key] for r in runs], dtype=float)
            row[key] = round(float(vals.mean()), 4)
            if len(runs) > 1 and key in ("b3_f1", "v_f1", "ari"):
                row[key + "_std"] = round(float(vals.std(ddof=1)), 4)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--leaderboard", help="CSV to update, e.g. report/leaderboard.csv")
    args = ap.parse_args()

    card = json.load(open(os.path.join(args.run_dir, "card.json"), encoding="utf-8"))
    check_card(card)
    rows = summarize({**score_fewrel(args.run_dir), **score_redocred(args.run_dir)})
    if not rows:
        sys.exit("no prediction files found")
    for r in rows:
        r.update({k: card[k] for k in ("run", "author", "method", "supervision")})
    json.dump(rows, open(os.path.join(args.run_dir, "scores.json"), "w"), indent=1)

    for r in rows:
        main_keys = (["micro_p", "micro_r", "micro_f1", "macro_f1", "evid_f1", "detect_f1"] if "micro_f1" in r
                     else ["b3_f1", "v_f1", "ari"])
        shown = "  ".join(f"{k}={r[k]:.3f}" + (f"±{r[k + '_std']:.3f}" if k + "_std" in r else "")
                          for k in main_keys)
        print(f"{r['dataset']:15s} {r['setting']:8s} seeds={r['n_seeds']}  {shown}")

    if args.leaderboard:
        old = read_csv(args.leaderboard) if os.path.exists(args.leaderboard) else []
        old = [r for r in old if r["run"] != card["run"]]          # re-running a run replaces its rows
        all_rows = old + rows
        front = ["run", "author", "method", "supervision", "dataset", "setting", "n_seeds"]
        cols = front + sorted({k for r in all_rows for k in r} - set(front))
        os.makedirs(os.path.dirname(args.leaderboard) or ".", exist_ok=True)
        with open(args.leaderboard, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)
        print(f"updated {args.leaderboard}")


if __name__ == "__main__":
    main()
