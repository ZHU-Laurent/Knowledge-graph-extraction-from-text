"""Score one run folder with the shared metrics and (optionally) add it to the leaderboard.

A run folder holds a card.json and prediction files named by convention (see docs/collaboration.md):

    results/<run>/card.json
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
from kg_extraction.data.redocred import RedocredGold, relation_buckets, train_fact_names  # noqa: E402
from kg_extraction.evaluation.graph import graph_scores  # noqa: E402
from kg_extraction.evaluation.metrics import (NO_RELATION, bootstrap_f1, clustering_scores,  # noqa: E402
                                              doc_counts, fact_scores, ign_scores, prf, typing_scores)
from kg_extraction.graph.build import build_graph  # noqa: E402

CARD_FIELDS = ["run", "author", "method", "supervision", "training_data", "models", "description"]
# What to look at first (see docs/evaluation.md, "Reading the leaderboard").
# 1. Ranking: micro F1, and Ign F1 for comparison with published DocRED results.
# 2. Diagnosis: where the errors come from (precision or recall, detection or typing, rare relations).
KEY_METRICS = ["micro_f1", "ign_f1"]
DIAGNOSTIC_METRICS = ["micro_p", "micro_r", "detect_f1", "typing_f1", "macro_f1"]
ID_COLUMNS = ["run", "author", "method", "supervision", "dataset", "setting", "n_seeds", "test_status"]
LABELS = {"micro_f1": "F1", "ign_f1": "Ign F1", "micro_p": "P", "micro_r": "R",
          "detect_f1": "Detection F1", "typing_f1": "Typing F1", "macro_f1": "Macro F1"}
SUPERVISION = {"none", "labels_for_mapping", "supervised"}
# Optional card field. "final" only for the single test run of a method, after every choice was made on dev.
TEST_STATUS = {"provisional", "final"}
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
    if card.get("test_status", "provisional") not in TEST_STATUS:
        sys.exit(f"card.json test_status must be one of {sorted(TEST_STATUS)}")


def score_redocred(run_dir):
    """Besides overall micro/macro scores it reports:
    - ign_*      Ign precision and F1 (DocRED): correct facts already seen in train are not counted
    - evid_*     facts whose gold annotation lists evidence sentences only (Re-DocRED added many facts
                 without evidence; predictions of those facts are then neither hits nor errors)
    - recall and F1 on pairs that share a sentence vs. pairs that do not
    - macro F1 on frequent / medium / rare relations (by train frequency)
    - detection P/R/F1: does the pair have at least one relation?
    - typing P/R/F1: on pairs related in gold and predicted related, is the relation right?
    - B3 / V / ARI on single-label related pairs, if a clusters file is given."""
    groups, golds, buckets, train_names = defaultdict(list), {}, None, None
    for path, subset, seed in redocred_files(run_dir):
        gold = golds.setdefault(subset, RedocredGold(subset, REDOCRED))
        buckets = buckets or relation_buckets(REDOCRED)
        train_names = train_names or train_fact_names(REDOCRED)
        pred, scores = read_redocred_facts(path, subset, gold, buckets)
        suffix = "" if seed is None else f"_seed{seed}"

        s = fact_scores(gold.facts, pred)
        s["micro_f1_ci_low"], s["micro_f1_ci_high"] = bootstrap_f1(doc_counts(gold.facts, pred, sorted(gold.docs)))
        per_rel = s.pop("per_relation")
        ev = fact_scores(gold.facts, pred, ignore=gold.facts - gold.with_evidence)
        s.update({"evid_p": ev["micro_p"], "evid_r": ev["micro_r"], "evid_f1": ev["micro_f1"]})
        s["ign_p"], s["ign_f1"] = ign_scores(gold.facts, pred, lambda f: gold.in_train(f, train_names))
        s["typing_p"], s["typing_r"], s["typing_f1"] = typing_scores(gold.facts, pred)

        hits = {f for f in gold.facts if f in pred}
        for name, keep in (("intra", True), ("inter", False)):
            g = [f for f in gold.facts if gold.same_sentence(f[0], f[1], f[3]) == keep]
            p = [f for f in pred if f[2] != NO_RELATION and gold.same_sentence(f[0], f[1], f[3]) == keep]
            s[f"recall_{name}_sentence"] = sum(f in hits for f in g) / len(g) if g else 0.0
            s[f"f1_{name}_sentence"] = prf(sum(f in hits for f in g), len(p), len(g))[2]
        for b in ("frequent", "medium", "rare"):
            f1s = [v["f1"] for r, v in per_rel.items() if buckets.get(r) == b]
            s[f"macro_f1_{b}"] = float(np.mean(f1s)) if f1s else 0.0

        gold_pairs = set(gold.pair_labels)
        pred_pairs = {(d, h, t) for d, h, r, t in pred if r != NO_RELATION}
        s["detect_p"], s["detect_r"], s["detect_f1"] = prf(len(gold_pairs & pred_pairs), len(pred_pairs), len(gold_pairs))
        s.update(graph_metrics(gold, pred, scores))

        with open(os.path.join(run_dir, f"redocred_{subset}_per_relation{suffix}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relation", "bucket", "gold", "pred", "tp", "p", "r", "f1"])
            for r, v in sorted(per_rel.items(), key=lambda kv: -kv[1]["gold"]):
                w.writerow([r, buckets.get(r), v["gold"], v["pred"], v["tp"],
                            round(v["p"], 4), round(v["r"], 4), round(v["f1"], 4)])

        cl_path = os.path.join(run_dir, f"redocred_{subset}_clusters{suffix}.csv")
        if os.path.exists(cl_path):
            clusters = {(r["doc_id"], int(r["h"]), int(r["t"])): r["cluster"] for r in read_csv(cl_path)}
            keys = [k for k, labels in gold.pair_labels.items() if len(labels) == 1 and k in clusters]
            if keys:
                cs = clustering_scores([next(iter(gold.pair_labels[k])) for k in keys], [clusters[k] for k in keys])
                s.update({f"cl_{k}": v for k, v in cs.items()})
        groups[(f"redocred_{subset}", "main")].append(s)
    return dict(groups)


def redocred_files(run_dir):
    """(path, subset, seed) of every facts file; seed is None for a file without a _seed<N> suffix."""
    files = []
    for path in sorted(glob.glob(os.path.join(run_dir, "redocred_*_facts*.csv"))):
        m = re.match(r"redocred_(dev|dev_debug|test)_facts(?:_seed(\d+))?\.csv$", os.path.basename(path))
        if not m:
            sys.exit(f"bad file name: {path} (allowed: redocred_<dev|dev_debug|test>_facts[_seed<N>].csv)")
        files.append((path, m.group(1), None if m.group(2) is None else int(m.group(2))))
    return files


def read_redocred_facts(path, subset, gold, buckets):
    """Checked facts {(doc_id, h, r, t)} of a file, and their scores (optional `score` column, else None)."""
    pred, scores = set(), {}
    for row in read_csv(path):
        doc, h, r, t = row["doc_id"], int(row["h"]), row["r"], int(row["t"])
        if doc not in gold.docs:
            sys.exit(f"{path}: document {doc} is not in subset {subset}")
        if not (0 <= h < gold.n_entities(doc) and 0 <= t < gold.n_entities(doc)) or h == t:
            sys.exit(f"{path}: bad entity pair ({doc}, {h}, {t})")
        if r != NO_RELATION and r not in buckets:
            sys.exit(f"{path}: unknown relation {r}")
        pred.add((doc, h, r, t))
        if row.get("score") not in (None, ""):
            scores[(doc, h, r, t)] = float(row["score"])
    return pred, (scores if len(scores) == len(pred) else None)


def graph_metrics(gold, pred, scores):
    """Corpus graph of the predicted facts against the gold graph (src/kg_extraction/graph/build.py).

    Entities merge on their Wikidata item, dates and numbers on their text. Precision at N ranks edges by
    noisy-or confidence, so it is only computed when the facts file has a `score` column."""
    doc_ids = sorted(gold.docs)
    docs, index = [gold.docs[d] for d in doc_ids], {d: i for i, d in enumerate(doc_ids)}
    facts = {(index[d], h, t, r): (scores[(d, h, r, t)] if scores else 1.0)
             for d, h, r, t in pred if r != NO_RELATION}
    g = graph_scores(docs, build_graph(docs, facts), ranks=(100, 500, 1000) if scores else ())
    out = {"graph_p": g["precision"] / 100, "graph_r": g["recall"] / 100, "graph_f1": g["f1"] / 100}
    out.update({f"graph_{k}": v / 100 for k, v in g.items() if k.startswith(("precision_at", "precision_support"))})
    return out


def summarize(groups):
    rows = []
    for (dataset, setting), runs in sorted(groups.items()):
        row = {"dataset": dataset, "setting": setting, "n_seeds": len(runs)}
        for key in runs[0]:
            vals = np.array([r[key] for r in runs], dtype=float)
            row[key] = round(float(vals.mean()), 4)
            if len(runs) > 1 and key in ("graph_f1", *KEY_METRICS, *DIAGNOSTIC_METRICS):
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
    rows = summarize(score_redocred(args.run_dir))
    if not rows:
        sys.exit("no prediction files found")
    for r in rows:
        r.update({k: card[k] for k in ("run", "author", "method", "supervision")})
        # Test is scored once per method, with frozen settings; until then its scores are provisional.
        r["test_status"] = card.get("test_status", "provisional") if r["dataset"].endswith("_test") else ""
    json.dump(rows, open(os.path.join(args.run_dir, "scores.json"), "w"), indent=1)

    for r in rows:
        shown = "  ".join(f"{k}={r[k]:.3f}" + (f"±{r[k + '_std']:.3f}" if k + "_std" in r else "")
                          for k in KEY_METRICS + DIAGNOSTIC_METRICS)
        print(f"{r['dataset']:15s} {r['setting']:8s} seeds={r['n_seeds']}  {shown}")

    if args.leaderboard:
        old = read_csv(args.leaderboard) if os.path.exists(args.leaderboard) else []
        old = [r for r in old if r["run"] != card["run"]]          # re-running a run replaces its rows
        all_rows = old + rows
        # Identity, then the key metrics, then the diagnostic ones, then everything else alphabetically.
        front = ID_COLUMNS + KEY_METRICS + DIAGNOSTIC_METRICS
        cols = front + sorted({k for r in all_rows for k in r} - set(front))
        os.makedirs(os.path.dirname(args.leaderboard) or ".", exist_ok=True)
        with open(args.leaderboard, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)
        summary = os.path.splitext(args.leaderboard)[0] + ".md"
        with open(summary, "w", encoding="utf-8") as f:
            f.write(summary_markdown(all_rows))
        print(f"updated {args.leaderboard} and {summary}")


def summary_markdown(rows):
    """Readable view of the Linked-Re-DocRED rows: key metrics first, one table per subset, best F1 first."""
    lines = ["# Leaderboard: Linked-Re-DocRED relation extraction", "",
             "Generated by `scripts/evaluate.py` from `leaderboard.csv`. Scores in %. "
             "See `docs/evaluation.md`, \"Reading the leaderboard\".", "",
             "- **Key metrics (rank on these):** F1, then Ign F1.",
             "- **Diagnosis:** P and R; Detection F1 (is the pair related?); Typing F1 (right relation, "
             "on detected related pairs); Macro F1 (every relation weighs the same, so rare ones count).",
             "- **Graph F1:** the same facts merged into one corpus graph (entities merged on their Wikidata item).",
             "- With several seeds, a score reads mean ± standard deviation. The 95% interval of F1 comes from "
             "resampling documents (bootstrap): two methods whose intervals overlap may not really differ, "
             "check with `scripts/compare_runs.py`.",
             "- Choices are made on `dev`. `test` is reported once per method, with frozen settings: "
             "until then its rows are marked *provisional*.", ""]

    def cell(r, k):
        value = r.get(k)
        if value in (None, ""):
            return "n/a"
        text = f"{100 * float(value):.2f}"
        if r.get(k + "_std") not in (None, ""):
            text += f" ± {100 * float(r[k + '_std']):.2f}"
        return f"**{text}**" if k in KEY_METRICS else text

    for dataset in ("redocred_test", "redocred_dev", "redocred_dev_debug"):
        subset = [r for r in rows if r["dataset"] == dataset and r.get("micro_f1") not in (None, "")]
        if not subset:
            continue
        subset.sort(key=lambda r: -float(r["micro_f1"]))
        keys = KEY_METRICS + DIAGNOSTIC_METRICS
        test = dataset.endswith("_test")
        lines += [f"## {dataset.replace('redocred_', '')}", "",
                  "| Run | Supervision | Seeds | " + ("Status | " if test else "") + "**F1** | F1 95% interval | "
                  + " | ".join(f"**{LABELS[k]}**" if k in KEY_METRICS else LABELS[k] for k in keys[1:])
                  + " | Graph F1 |",
                  "|---|---|---|" + ("---|" if test else "") + "---:|---|" + "---:|" * len(keys)]
        for r in subset:
            ci = (f"{100 * float(r['micro_f1_ci_low']):.1f} to {100 * float(r['micro_f1_ci_high']):.1f}"
                  if r.get("micro_f1_ci_low") not in (None, "") else "n/a")
            status = f"*{r.get('test_status') or 'provisional'}* | " if test else ""
            lines.append(f"| {r['run']} | {r['supervision']} | {r['n_seeds']} | {status}{cell(r, keys[0])} | {ci} | "
                         + " | ".join(cell(r, k) for k in keys[1:]) + f" | {cell(r, 'graph_f1')} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
