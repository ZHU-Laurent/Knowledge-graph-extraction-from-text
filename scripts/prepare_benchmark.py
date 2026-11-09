"""Write the frozen FewRel NYT / PubMed benchmark files that every method must use.

    python scripts/prepare_benchmark.py
    -> data/processed/benchmark/{nyt,pubmed}.jsonl       inputs with stable ids, no labels
    -> data/processed/benchmark/{nyt,pubmed}_gold.csv    id,relation (only read by scripts/evaluate.py)

Linked-Re-DocRED: put train_revised.json, dev_revised.json and test_revised.json in data/raw/Linked-Re-DocRED/
(https://github.com/alteca/Linked-DocRED/tree/main/Linked-Re-DocRED). The script then writes, for each fixed
subset (train_debug, dev_debug, train, dev, test):
-> redocred_<subset>_docs.jsonl    text and entities, no relations
-> redocred_<subset>_pairs.csv     every ordered entity pair, with entity types and whether they share a sentence
-> redocred_subsets.json           the document ids of each subset (seed 0)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_extraction.data.benchmark import prepare_fewrel  # noqa: E402
from kg_extraction.data.redocred import RAW_DIR, prepare_redocred  # noqa: E402

if __name__ == "__main__":
    prepare_fewrel()
    print("FewRel NYT / PubMed: done")
    if os.path.exists(os.path.join(RAW_DIR, "train_revised.json")):
        for subset, info in prepare_redocred().items():
            print(f"Linked-Re-DocRED {subset:12s} {info['n_docs']:5d} documents {info['n_pairs']:9d} ordered pairs")
    else:
        print(f"Linked-Re-DocRED skipped: put train/dev/test_revised.json in {RAW_DIR}/")
