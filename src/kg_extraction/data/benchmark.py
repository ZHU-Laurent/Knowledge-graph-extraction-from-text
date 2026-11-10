"""Frozen benchmark data with stable instance ids, shared by every method.

FewRel NYT / PubMed (clustering track): one instance per (sentence, head, tail). The id is the position in
the official file order, so it does not reveal the relation. Gold labels are kept in a separate file.

Linked-Re-DocRED (main track) lives in kg_extraction.data.redocred.
"""
import csv
import json
import os
import urllib.request

FEWREL_URL = "https://raw.githubusercontent.com/thunlp/FewRel/master/data/"
FEWREL_FILES = {"nyt": "val_nyt.json", "pubmed": "val_pubmed.json"}


def fewrel_instances(path):
    """Yield (instance, relation) in official file order. Uses the first mention of each entity."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for relation, items in data.items():
        for item in items:
            h_pos, t_pos = item["h"][2][0], item["t"][2][0]
            yield {"tokens": item["tokens"],
                   "h": {"name": item["h"][0], "start": h_pos[0], "end": h_pos[-1] + 1},
                   "t": {"name": item["t"][0], "start": t_pos[0], "end": t_pos[-1] + 1}}, relation


def prepare_fewrel(raw_dir="data/raw/fewrel", out_dir="data/processed/benchmark", names=("nyt", "pubmed")):
    """Download FewRel NYT / PubMed if needed, then write <name>.jsonl (inputs, no labels)
    and <name>_gold.csv (id, relation)."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        raw = os.path.join(raw_dir, FEWREL_FILES[name])
        if not os.path.exists(raw):
            urllib.request.urlretrieve(FEWREL_URL + FEWREL_FILES[name], raw)
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as fx, \
                open(os.path.join(out_dir, f"{name}_gold.csv"), "w", newline="", encoding="utf-8") as fy:
            gold = csv.writer(fy)
            gold.writerow(["id", "relation"])
            for i, (inst, relation) in enumerate(fewrel_instances(raw)):
                fx.write(json.dumps({"id": f"{name}-{i:05d}", **inst}) + "\n")
                gold.writerow([f"{name}-{i:05d}", relation])


def read_fewrel_gold(path):
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["relation"] for row in csv.DictReader(f)}

