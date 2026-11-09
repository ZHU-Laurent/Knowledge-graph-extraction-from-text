"""DocRED relation metrics with gold entities: F1, Ign F1, intra / inter F1 and detection F1.

A fact is a (document index, head id, tail id, relation code) tuple. Ign F1 follows
the official DocRED evaluation: correct facts whose mention names already form a
train fact are removed from the precision, and the recall is left unchanged.
"""
from kg_extraction.relations.prompts import shares_sentence


def gold_facts(docs):
    """Return the set of gold facts of a list of documents."""
    return {(index, relation["h"], relation["t"], relation["r"])
            for index, doc in enumerate(docs) for relation in doc["relations"]}


def train_fact_names(train_docs):
    """Return the (head name, tail name, relation) triples of every train fact mention pair."""
    names = set()
    for doc in train_docs:
        entities = doc["entities"]
        for relation in doc["relations"]:
            for head in entities[relation["h"]]["mentions"]:
                for tail in entities[relation["t"]]["mentions"]:
                    names.add((head["name"], tail["name"], relation["r"]))
    return names


def _prf(n_correct, n_predicted, n_gold):
    precision = n_correct / n_predicted if n_predicted else 0.0
    recall = n_correct / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _in_train(docs, fact, train_names):
    index, head, tail, relation = fact
    entities = docs[index]["entities"]
    return any((first["name"], second["name"], relation) in train_names
               for first in entities[head]["mentions"]
               for second in entities[tail]["mentions"])


def relation_scores(docs, predicted, train_names):
    """Return the DocRED metrics of predicted facts, as percentages."""
    gold = gold_facts(docs)
    predicted = set(predicted)
    correct = predicted & gold
    precision, recall, f1 = _prf(len(correct), len(predicted), len(gold))

    correct_in_train = sum(_in_train(docs, fact, train_names) for fact in correct)
    ign_precision = ((len(correct) - correct_in_train) / (len(predicted) - correct_in_train)
                     if len(predicted) > correct_in_train else 0.0)
    ign_f1 = (2 * ign_precision * recall / (ign_precision + recall)
              if ign_precision + recall else 0.0)

    def shared(fact):
        entities = docs[fact[0]]["entities"]
        return shares_sentence(entities[fact[1]], entities[fact[2]])

    scores = {"precision": precision, "recall": recall, "f1": f1, "ign_f1": ign_f1}
    for name, keep in (("intra", True), ("inter", False)):
        subset_gold = {fact for fact in gold if shared(fact) == keep}
        subset_predicted = {fact for fact in predicted if shared(fact) == keep}
        scores[f"{name}_f1"] = _prf(len(subset_predicted & subset_gold),
                                    len(subset_predicted), len(subset_gold))[2]

    gold_pairs = {fact[:3] for fact in gold}
    predicted_pairs = {fact[:3] for fact in predicted}
    detection = _prf(len(predicted_pairs & gold_pairs), len(predicted_pairs), len(gold_pairs))
    scores["detection_precision"], scores["detection_recall"], scores["detection_f1"] = detection
    scores = {name: round(100 * value, 2) for name, value in scores.items()}
    scores.update({"predicted_facts": len(predicted), "gold_facts": len(gold)})
    return scores
