"""Write and read the prediction files that scripts/evaluate.py scores."""
import csv
import json
from pathlib import Path

SUBSETS_MANIFEST = "data/processed/benchmark/redocred_subsets.json"


def write_shared_run(directory, docs, facts, card, seed):
    """Write one run folder for the shared evaluator: the facts of one seed, and the card.

    facts maps "dev" and, optionally, "test" to {(doc index, head, tail, relation): score}.
    dev facts are also written for dev_debug.
    """
    directory.mkdir(parents=True, exist_ok=True)
    debug_ids = set(json.loads(Path(SUBSETS_MANIFEST).read_text())["dev_debug"]["doc_ids"])
    subsets = {"dev_debug": ("dev", debug_ids), "dev": ("dev", None), "test": ("test", None)}
    for subset, (split, kept) in subsets.items():
        if split not in facts:
            continue
        rows = sorted((str(docs[split][index]["id"]), head, relation, tail, round(score, 6))
                      for (index, head, tail, relation), score in facts[split].items())
        rows = [row for row in rows if kept is None or row[0] in kept]
        with open(directory / f"redocred_{subset}_facts_seed{seed}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["doc_id", "h", "r", "t", "score"])
            writer.writerows(rows)
    (directory / "card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")


def read_run_facts(directory, subset, seed, docs):
    """Return {(document index, head id, tail id, relation code): score} kept by a run.

    docs are the documents of the split that the subset comes from.
    """
    index_of = {str(doc["id"]): index for index, doc in enumerate(docs)}
    with open(Path(directory) / f"redocred_{subset}_facts_seed{seed}.csv", newline="") as file:
        return {(index_of[row["doc_id"]], int(row["h"]), int(row["t"]), row["r"]):
                float(row["score"]) for row in csv.DictReader(file)}
