"""Relation detector and typing classifiers on frozen PromptORE embeddings (method M2).

The detector predicts whether a pair has at least one relation. The typing
classifiers, one per relation, are trained on related pairs only. The score of a
fact is P(related) x P(relation | related).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kg_extraction.pairs.candidates import candidate_pairs, gold_labels
from kg_extraction.relations.prompts import build_context, entity_name

CONTEXT = "mention_sentences"


def pair_list(docs, allowed):
    """Return the candidate pairs of documents.

    Each pair is (document index, head id, tail id, gold relation codes). Predicted
    documents have no relations, so their pairs have no codes.
    """
    pairs = []
    for index, doc in enumerate(docs):
        labels = gold_labels(doc)
        for head, tail in candidate_pairs(doc, allowed):
            pairs.append((index, head["id"], tail["id"],
                          sorted(labels.get((head["id"], tail["id"]), ()))))
    return pairs


def pair_instances(docs, pairs, context=CONTEXT):
    """Return the PromptORE instance of each pair."""
    instances = []
    for index, head, tail, _ in pairs:
        entities = docs[index]["entities"]
        instances.append({"context": build_context(docs[index], entities[head], entities[tail],
                                                   context),
                          "head": entity_name(entities[head]),
                          "tail": entity_name(entities[tail])})
    return instances


def sample_train_pairs(pairs, negatives_per_positive, seed):
    """Return the indices of all related pairs and of a random sample of unrelated pairs."""
    related = [index for index, pair in enumerate(pairs) if pair[3]]
    unrelated = [index for index, pair in enumerate(pairs) if not pair[3]]
    n_sampled = min(len(unrelated), negatives_per_positive * len(related))
    sampled = np.random.default_rng(seed).choice(unrelated, size=n_sampled, replace=False)
    return sorted(related + sampled.tolist()), len(unrelated) / max(n_sampled, 1)


class RelationProbe:
    def __init__(self, relations, max_iter=1000, n_jobs=-1):
        self.relations = list(relations)
        self.detector = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
        self.typing = make_pipeline(StandardScaler(), OneVsRestClassifier(
            LogisticRegression(max_iter=max_iter), n_jobs=n_jobs))

    def fit(self, embeddings, labels, negative_weight=1.0):
        """Fit on embeddings and their gold relation codes.

        negative_weight gives each sampled unrelated pair the weight of the unrelated
        pairs it stands for, so the detector learns the true share of related pairs.
        """
        related = np.array([bool(codes) for codes in labels])
        weights = np.where(related, 1.0, negative_weight)
        self.detector.fit(embeddings, related, logisticregression__sample_weight=weights)
        position = {code: column for column, code in enumerate(self.relations)}
        targets = np.zeros((related.sum(), len(self.relations)), dtype=int)
        for row, codes in enumerate(codes for codes in labels if codes):
            for code in codes:
                targets[row, position[code]] = 1
        self.typing.fit(embeddings[related], targets)
        return self

    def scores(self, embeddings):
        """Return the (pairs, relations) matrix of fact scores."""
        related = self.detector.predict_proba(embeddings)[:, 1]
        return related[:, None] * self.typing.predict_proba(embeddings)


def facts_above(pairs, scores, relations, threshold):
    """Return {fact: score} for every (pair, relation) whose score passes the threshold."""
    rows, columns = np.nonzero(scores >= threshold)
    return {(pairs[row][0], pairs[row][1], pairs[row][2], relations[column]):
            float(scores[row, column]) for row, column in zip(rows, columns)}


def best_threshold(pairs, scores, relations, gold, candidates=None):
    """Return the threshold that maximizes the micro F1 against a set of gold facts."""
    candidates = candidates if candidates is not None else np.linspace(0.05, 0.95, 91)
    best = (0.0, 0.5)
    for threshold in candidates:
        predicted = facts_above(pairs, scores, relations, threshold)
        correct = len(gold & predicted.keys())
        if correct:
            f1 = 2 * correct / (len(predicted) + len(gold))
            best = max(best, (f1, float(threshold)))
    return best[1]
