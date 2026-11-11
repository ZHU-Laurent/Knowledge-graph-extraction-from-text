"""Linked-Re-DocRED benchmark: fixed subsets, label-free inputs, gold facts.

Subsets are fixed once (seed 0) so that every member scores on exactly the same documents:

    train_debug  300 random train documents     quick experiments, fitting on a small budget
    dev_debug    100 random dev documents       quick tuning
    train / dev / test                          the official splits, complete

Methods read data/processed/benchmark/redocred_<subset>_docs.jsonl (text and entities, no relations) and
redocred_<subset>_pairs.csv (every ordered entity pair). Labels come from the raw files:
train labels may be used by `labels_for_mapping` and `supervised` runs, dev labels for tuning,
test labels only by scripts/evaluate.py.
"""
import csv
import json
import os
import random
from collections import Counter

RAW_DIR = "data/raw/Linked-Re-DocRED"
SPLIT_FILES = {"train": "train_revised.json", "dev": "dev_revised.json", "test": "test_revised.json"}
SUBSETS = {"train_debug": ("train", 300), "dev_debug": ("dev", 100),
           "train": ("train", None), "dev": ("dev", None), "test": ("test", None)}
SEED = 0


def load_split(split, raw_dir=RAW_DIR):
    with open(os.path.join(raw_dir, SPLIT_FILES[split]), encoding="utf-8") as f:
        return json.load(f)


def subset_doc_ids(docs, size, seed=SEED):
    ids = sorted(str(d["id"]) for d in docs)
    return ids if size is None or size >= len(ids) else sorted(random.Random(seed).sample(ids, size))


def entity_name(entity):
    """Longest mention, as in the project plan."""
    return max((m["name"] for m in entity["mentions"]), key=len)


def same_sentence(doc, h, t):
    sh = {m["sent_id"] for m in doc["entities"][h]["mentions"]}
    st = {m["sent_id"] for m in doc["entities"][t]["mentions"]}
    return bool(sh & st)


def prepare_redocred(raw_dir=RAW_DIR, out_dir="data/processed/benchmark"):
    os.makedirs(out_dir, exist_ok=True)
    cache, manifest = {}, {}
    for subset, (split, size) in SUBSETS.items():
        docs = cache.setdefault(split, load_split(split, raw_dir))
        ids = set(subset_doc_ids(docs, size))
        chosen = [d for d in docs if str(d["id"]) in ids]
        manifest[subset] = {"split": split, "seed": SEED, "n_docs": len(chosen), "doc_ids": sorted(ids)}
        n_pairs = 0
        with open(os.path.join(out_dir, f"redocred_{subset}_docs.jsonl"), "w", encoding="utf-8") as fd, \
                open(os.path.join(out_dir, f"redocred_{subset}_pairs.csv"), "w", newline="", encoding="utf-8") as fp:
            pw = csv.writer(fp)
            pw.writerow(["doc_id", "h", "t", "h_type", "t_type", "same_sentence"])
            for d in chosen:
                ents = [{"id": i, "type": e["type"], "name": entity_name(e),
                         "wikidata": e.get("entity_linking", {}).get("wikidata_resource"),
                         "mentions": [{"name": m["name"], "sent_id": m["sent_id"], "pos": m["pos"]}
                                      for m in e["mentions"]]}
                        for i, e in enumerate(d["entities"])]
                fd.write(json.dumps({"doc_id": str(d["id"]), "title": d["title"], "sents": d["sents"],
                                     "entities": ents}, ensure_ascii=False) + "\n")
                for h in range(len(ents)):
                    for t in range(len(ents)):
                        if h != t:
                            pw.writerow([d["id"], h, t, ents[h]["type"], ents[t]["type"], int(same_sentence(d, h, t))])
                            n_pairs += 1
        manifest[subset]["n_pairs"] = n_pairs
    with open(os.path.join(out_dir, "redocred_subsets.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


class RedocredGold:
    """Gold facts of one subset, with the metadata the evaluator needs for its breakdowns."""

    def __init__(self, subset, raw_dir=RAW_DIR, manifest_path="data/processed/benchmark/redocred_subsets.json"):
        with open(manifest_path) as f:
            info = json.load(f)[subset]
        ids = set(info["doc_ids"])
        self.docs = {str(d["id"]): d for d in load_split(info["split"], raw_dir) if str(d["id"]) in ids}
        self.facts, self.with_evidence, self.pair_labels = set(), set(), {}
        for doc_id, d in self.docs.items():
            for r in d["relations"]:
                fact = (doc_id, int(r["h"]), r["r"], int(r["t"]))
                self.facts.add(fact)
                if r.get("evidence"):
                    self.with_evidence.add(fact)
                self.pair_labels.setdefault((doc_id, fact[1], fact[3]), set()).add(r["r"])

    def n_entities(self, doc_id):
        return len(self.docs[doc_id]["entities"])

    def same_sentence(self, doc_id, h, t):
        return same_sentence(self.docs[doc_id], h, t)

    def in_train(self, fact, names):
        """True if some mention pair of the fact's entities forms a train fact (`names`, see train_fact_names)."""
        doc_id, h, r, t = fact
        entities = self.docs[doc_id]["entities"]
        return any((mh["name"], mt["name"], r) in names
                   for mh in entities[h]["mentions"] for mt in entities[t]["mentions"])


def train_fact_names(raw_dir=RAW_DIR):
    """(head mention name, tail mention name, relation) of every train fact, for Ign F1 (DocRED)."""
    names = set()
    for d in load_split("train", raw_dir):
        for r in d["relations"]:
            for mh in d["entities"][r["h"]]["mentions"]:
                for mt in d["entities"][r["t"]]["mentions"]:
                    names.add((mh["name"], mt["name"], r["r"]))
    return names


def relation_buckets(raw_dir=RAW_DIR, n_frequent=10, rare_below=200):
    """frequent = 10 most common train relations, rare = fewer than 200 train facts, medium = the rest."""
    counts = Counter(r["r"] for d in load_split("train", raw_dir) for r in d["relations"])
    ranked = [r for r, _ in counts.most_common()]
    return {r: "frequent" if i < n_frequent else ("rare" if counts[r] < rare_below else "medium")
            for i, r in enumerate(ranked)}
