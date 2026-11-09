"""Write the frozen FewRel NYT / PubMed benchmark files that every method must use.

    python scripts/prepare_benchmark.py
    -> data/processed/benchmark/{nyt,pubmed}.jsonl       inputs with stable ids, no labels
    -> data/processed/benchmark/{nyt,pubmed}_gold.csv    id,relation (only read by scripts/evaluate.py)

Linked-Re-DocRED needs no conversion: put dev_revised.json and test_revised.json in data/raw/Linked-Re-DocRED/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_extraction.data.benchmark import prepare_fewrel  # noqa: E402

if __name__ == "__main__":
    prepare_fewrel()
    print("wrote data/processed/benchmark/")
