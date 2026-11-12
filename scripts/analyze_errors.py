"""Error analysis of run folders on a Linked-Re-DocRED subset (experiment D of the project plan).

For each run (one seed), it splits the errors into detection and typing errors, and breaks the recall down by
sentence distance, entity type pair, number of labels of the pair and relation; it also checks calibration
(precision by score) and how often a false positive is the inverse of a gold fact or a fact gold elsewhere in
the corpus. Run it on dev: test is kept for the final scores.

Usage:
    python scripts/analyze_errors.py results/atlop-frozen results/promptore-classifier --subset dev --out results/analysis
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import REDOCRED, read_redocred_facts, redocred_files  # noqa: E402
from kg_extraction.data.redocred import RedocredGold, relation_buckets  # noqa: E402
from kg_extraction.graph.build import node_key  # noqa: E402

INVERSE = {"P150": "P131", "P131": "P150", "P527": "P361", "P361": "P527", "P355": "P749", "P749": "P355",
           "P155": "P156", "P156": "P155", "P40": "P22", "P22": "P40", "P25": "P40", "P1441": "P674", "P674": "P1441"}
SYMMETRIC = {"P26", "P3373", "P47", "P530"}


def sentence_distance(doc, h, t):
    sh = {m["sent_id"] for m in doc["entities"][h]["mentions"]}
    st = {m["sent_id"] for m in doc["entities"][t]["mentions"]}
    return min(abs(a - b) for a in sh for b in st)


def ratio(a, b):
    return round(100 * a / b, 2) if b else None


def analyze(gold, pred, scores, buckets):
    docs = gold.docs
    gold_facts, pred = gold.facts, {f for f in pred if f[2] != "no_relation"}
    gold_pairs = defaultdict(set)
    for d, h, r, t in gold_facts:
        gold_pairs[(d, h, t)].add(r)
    pred_pairs = defaultdict(set)
    for d, h, r, t in pred:
        pred_pairs[(d, h, t)].add(r)
    hits = gold_facts & pred

    # Where the missed facts and the false facts come from
    fn = gold_facts - pred
    fp = pred - gold_facts
    out = {"n_gold": len(gold_facts), "n_pred": len(pred), "n_hits": len(hits),
           "missed": {"total": len(fn),
                      "pair_not_found": sum((d, h, t) not in pred_pairs for d, h, _, t in fn),
                      "pair_found_relation_missing": sum((d, h, t) in pred_pairs for d, h, _, t in fn)},
           "false": {"total": len(fp),
                     "unrelated_pair": sum((d, h, t) not in gold_pairs for d, h, _, t in fp),
                     "related_pair_wrong_relation": sum((d, h, t) in gold_pairs for d, h, _, t in fp)}}

    # A false fact may be a fact the annotators left out: inverse or symmetric of a gold fact,
    # or the same (head node, relation, tail node) gold in another document of the subset.
    keys = {d: [node_key(doc, {"id": i, **e}) for i, e in enumerate(doc["entities"])] for d, doc in docs.items()}
    corpus = {(keys[d][h], r, keys[d][t]) for d, h, r, t in gold_facts}
    inverse = sum((d, t, INVERSE[r], h) in gold_facts for d, h, r, t in fp if r in INVERSE)
    symmetric = sum((d, t, r, h) in gold_facts for d, h, r, t in fp if r in SYMMETRIC)
    elsewhere = sum((keys[d][h], r, keys[d][t]) in corpus and not keys[d][h].startswith("doc:")
                    for d, h, r, t in fp)
    out["false"].update({"inverse_of_gold": inverse, "symmetric_of_gold": symmetric,
                         "gold_in_another_document": elsewhere})

    def recall_by(key_fn):
        total, found = Counter(), Counter()
        for f in gold_facts:
            k = key_fn(f)
            total[k] += 1
            found[k] += f in hits
        return {str(k): {"gold": total[k], "recall": ratio(found[k], total[k])} for k in sorted(total, key=str)}

    def precision_by(key_fn):
        total, right = Counter(), Counter()
        for f in pred:
            k = key_fn(f)
            total[k] += 1
            right[k] += f in hits
        return {str(k): {"pred": total[k], "precision": ratio(right[k], total[k])} for k in sorted(total, key=str)}

    distance = lambda f: min(sentence_distance(docs[f[0]], f[1], f[3]), 3)
    out["recall_by_sentence_distance"] = recall_by(distance)
    out["precision_by_sentence_distance"] = precision_by(distance)
    out["recall_by_labels_of_pair"] = recall_by(lambda f: min(len(gold_pairs[(f[0], f[1], f[3])]), 3))
    out["recall_by_bucket"] = recall_by(lambda f: buckets.get(f[2], "unseen"))
    types = lambda f: f"{docs[f[0]]['entities'][f[1]]['type']}-{docs[f[0]]['entities'][f[3]]['type']}"
    by_types = recall_by(types)
    out["recall_by_type_pair"] = dict(sorted(by_types.items(), key=lambda kv: -kv[1]["gold"])[:10])
    n_entities = lambda f: "<15" if len(docs[f[0]]["entities"]) < 15 else ("15-24" if len(docs[f[0]]["entities"]) < 25 else "25+")
    out["recall_by_entities_in_document"] = recall_by(n_entities)

    per_relation = Counter(f[2] for f in fn)
    out["most_missed_relations"] = per_relation.most_common(10)
    out["most_false_relations"] = Counter(f[2] for f in fp).most_common(10)

    if scores:
        bins = defaultdict(lambda: [0, 0])
        for f in pred:
            b = min(int(scores[f] * 10), 9) / 10
            bins[b][0] += 1
            bins[b][1] += f in hits
        out["precision_by_score"] = {f"{b:.1f}-{b + 0.1:.1f}": {"pred": n, "precision": ratio(k, n)}
                                     for b, (n, k) in sorted(bins.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--subset", default="dev", choices=["dev_debug", "dev"])
    ap.add_argument("--seed", type=int, default=0, help="seed file to analyze, when a run has several")
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()

    gold, buckets = RedocredGold(args.subset, REDOCRED), relation_buckets(REDOCRED)
    report, preds = {}, {}
    for run in args.runs:
        files = [(p, s) for p, sub, s in redocred_files(run) if sub == args.subset]
        path = next((p for p, s in files if s in (args.seed, None)), None)
        if path is None:
            sys.exit(f"{run}: no {args.subset} facts for seed {args.seed}")
        pred, scores = read_redocred_facts(path, args.subset, gold, buckets)
        name = os.path.basename(os.path.normpath(run))
        preds[name] = {f for f in pred if f[2] != "no_relation"}
        report[name] = analyze(gold, pred, scores, buckets)

    # How complementary the methods are: facts found by one and not the other, and the union
    names = list(preds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            hits_a, hits_b = preds[a] & gold.facts, preds[b] & gold.facts
            report[f"{a} vs {b}"] = {"hits_only_" + a: len(hits_a - hits_b), "hits_only_" + b: len(hits_b - hits_a),
                                     "hits_both": len(hits_a & hits_b),
                                     "recall_of_union": ratio(len(hits_a | hits_b), len(gold.facts))}

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"errors_{args.subset}_seed{args.seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
