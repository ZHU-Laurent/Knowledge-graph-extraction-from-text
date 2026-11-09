"""Build candidate ordered entity pairs and filter them by entity types."""
from collections import defaultdict


def gold_labels(doc):
    """Return the relation codes of each related (head id, tail id) pair."""
    labels = defaultdict(set)
    for relation in doc["relations"]:
        labels[(relation["h"], relation["t"])].add(relation["r"])
    return dict(labels)


def allowed_type_pairs(docs):
    """Return the (head type, tail type) combinations that carry a relation."""
    allowed = set()
    for doc in docs:
        types = {entity["id"]: entity["type"] for entity in doc["entities"]}
        for head, tail in gold_labels(doc):
            allowed.add((types[head], types[tail]))
    return allowed


def candidate_pairs(doc, allowed=None):
    """Yield the ordered entity pairs of a document, optionally filtered by type."""
    for head in doc["entities"]:
        for tail in doc["entities"]:
            if head["id"] == tail["id"]:
                continue
            if allowed is None or (head["type"], tail["type"]) in allowed:
                yield head, tail


def filter_report(docs, allowed):
    """Count what the type filter keeps and loses on a set of documents."""
    report = {"pairs_before": 0, "pairs_after": 0, "related_pairs": 0, "related_pairs_lost": 0}
    for doc in docs:
        labels = gold_labels(doc)
        n_entities = len(doc["entities"])
        report["pairs_before"] += n_entities * (n_entities - 1)
        kept = {(head["id"], tail["id"]) for head, tail in candidate_pairs(doc, allowed)}
        report["pairs_after"] += len(kept)
        report["related_pairs"] += len(labels)
        report["related_pairs_lost"] += sum(pair not in kept for pair in labels)
    return report


def related_pairs(doc):
    """Yield (head, tail, labels) for each related pair of a document."""
    entities = {entity["id"]: entity for entity in doc["entities"]}
    for (head, tail), labels in gold_labels(doc).items():
        yield entities[head], entities[tail], labels
