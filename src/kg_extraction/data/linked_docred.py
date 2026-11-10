"""Load Linked-Re-DocRED documents."""
import json
from pathlib import Path

SPLIT_FILES = {
    "train": "train_revised.json",
    "dev": "dev_revised.json",
    "test": "test_revised.json",
}


def load_split(data_dir, split):
    """Return the list of documents of a split."""
    path = Path(data_dir) / SPLIT_FILES[split]
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_relation_names(data_dir):
    """Return the mapping from Wikidata relation code to relation name."""
    with open(Path(data_dir) / "rel_info.json", encoding="utf-8") as file:
        return json.load(file)
