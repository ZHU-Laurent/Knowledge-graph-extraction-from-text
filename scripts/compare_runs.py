"""Is run A really better than run B? Paired bootstrap over documents on the micro F1 of Linked-Re-DocRED.

Both runs are scored on the same resampled documents, so the test only keeps the noise that comes from which
documents are in the subset. With several seeds, the counts of all seeds are pooled.

Usage:
    python scripts/compare_runs.py results/<run A> results/<run B> --subset test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import REDOCRED, read_redocred_facts, redocred_files  # noqa: E402
from kg_extraction.data.redocred import RedocredGold, relation_buckets  # noqa: E402
from kg_extraction.evaluation.metrics import doc_counts, paired_bootstrap  # noqa: E402


def run_counts(run_dir, subset, gold, buckets):
    paths = [path for path, s, _ in redocred_files(run_dir) if s == subset]
    if not paths:
        sys.exit(f"{run_dir}: no redocred_{subset}_facts file")
    doc_ids = sorted(gold.docs)
    return sum(doc_counts(gold.facts, read_redocred_facts(path, subset, gold, buckets)[0], doc_ids)
               for path in paths), len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--subset", default="dev", choices=["dev_debug", "dev", "test"])
    ap.add_argument("--samples", type=int, default=1000)
    args = ap.parse_args()

    gold, buckets = RedocredGold(args.subset, REDOCRED), relation_buckets(REDOCRED)
    (a, seeds_a), (b, seeds_b) = (run_counts(r, args.subset, gold, buckets) for r in (args.run_a, args.run_b))
    delta, low, high, p = paired_bootstrap(a, b, n_samples=args.samples)
    name_a, name_b = os.path.basename(os.path.normpath(args.run_a)), os.path.basename(os.path.normpath(args.run_b))
    print(f"{args.subset}: F1({name_a}, {seeds_a} seed(s)) - F1({name_b}, {seeds_b} seed(s)) = {100 * delta:+.2f} points, "
          f"95% interval [{100 * low:+.2f}, {100 * high:+.2f}], p = {p:.3f}")
    print(f"{name_a} is better beyond document sampling noise" if p < 0.05 else
          "not significant at 5%: the gap may come from which documents are in the subset")


if __name__ == "__main__":
    main()
