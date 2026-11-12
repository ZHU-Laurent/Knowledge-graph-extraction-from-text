"""Write the frozen Linked-Re-DocRED benchmark files that every method must use.

Put train_revised.json, dev_revised.json and test_revised.json in data/raw/Linked-Re-DocRED/
(https://github.com/alteca/Linked-DocRED/tree/main/Linked-Re-DocRED), then run:

    python scripts/prepare_benchmark.py

For each fixed subset (train_debug, dev_debug, train, dev, test) it writes, in data/processed/benchmark/:
-> redocred_<subset>_docs.jsonl    text and entities, no relations
-> redocred_<subset>_pairs.csv     every ordered entity pair, with entity types and whether they share a sentence
-> redocred_subsets.json           the document ids of each subset (seed 0)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_extraction.data.redocred import RAW_DIR, prepare_redocred  # noqa: E402

if __name__ == "__main__":
    if not os.path.exists(os.path.join(RAW_DIR, "train_revised.json")):
        sys.exit(f"put train/dev/test_revised.json in {RAW_DIR}/")
    for subset, info in prepare_redocred().items():
        print(f"Linked-Re-DocRED {subset:12s} {info['n_docs']:5d} documents {info['n_pairs']:9d} ordered pairs")
