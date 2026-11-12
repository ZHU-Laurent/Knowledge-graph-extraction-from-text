"""Shared metrics. Every method in the project is scored with these functions, so results are comparable.

Clustering metrics compare predicted cluster ids with gold relations (PromptORE paper, section 4.1).
Fact metrics compare predicted (document, head, relation, tail) facts with gold facts.
"""
from collections import Counter

import numpy as np
from sklearn.metrics.cluster import (adjusted_rand_score, completeness_score,
                                     contingency_matrix, homogeneity_score)

NO_RELATION = "no_relation"


def bcubed(gold, pred):
    """B3 precision, recall and F1 (Bagga and Baldwin 1998)."""
    c = contingency_matrix(gold, pred)
    w = c / c.sum()
    precision = float(np.sum(w * (c / c.sum(axis=0))))
    recall = float(np.sum(w * (c / c.sum(axis=1, keepdims=True))))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def clustering_scores(gold, pred):
    """B3, V-measure and ARI. `gold` and `pred` are aligned label sequences of any hashable type."""
    gold, pred = list(gold), list(pred)
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")
    b3_p, b3_r, b3_f1 = bcubed(gold, pred)
    hom, comp = homogeneity_score(gold, pred), completeness_score(gold, pred)
    v_f1 = 2 * hom * comp / (hom + comp) if hom + comp else 0.0
    return {"b3_p": b3_p, "b3_r": b3_r, "b3_f1": b3_f1,
            "v_hom": hom, "v_comp": comp, "v_f1": v_f1,
            "ari": adjusted_rand_score(gold, pred),
            "n_instances": len(gold), "n_clusters": len(set(pred))}


def fact_scores(gold_facts, pred_facts, ignore=()):
    """Precision, recall and F1 on facts, excluding "no relation".

    Facts are (doc_id, head, relation, tail) tuples. Duplicates are ignored. With gold entities, this
    should equal the micro Relation F1 of Linked-DocRED (metrics/relation_f1.py, hard aggregation);
    the project plan still cross-checks the two once.
    Facts in `ignore` are removed from both sides (used to score a subset of the gold facts).
    Returns micro scores, macro scores over gold relation types, and per-relation counts.
    """
    ignore = set(ignore)                      # facts that count neither as hits nor as errors
    gold = {f for f in gold_facts if f[2] != NO_RELATION} - ignore
    pred = {f for f in pred_facts if f[2] != NO_RELATION} - ignore
    tp = gold & pred
    micro_p = len(tp) / len(pred) if pred else 0.0
    micro_r = len(tp) / len(gold) if gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    n_gold, n_pred, n_tp = Counter(f[2] for f in gold), Counter(f[2] for f in pred), Counter(f[2] for f in tp)
    per_relation, f1s = {}, []
    for r in sorted(n_gold):
        p = n_tp[r] / n_pred[r] if n_pred[r] else 0.0
        rec = n_tp[r] / n_gold[r]
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        per_relation[r] = {"gold": n_gold[r], "pred": n_pred[r], "tp": n_tp[r], "p": p, "r": rec, "f1": f1}
        f1s.append(f1)
    return {"micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "n_gold": len(gold), "n_pred": len(pred), "n_tp": len(tp),
            "per_relation": per_relation}


def prf(n_tp, n_pred, n_gold):
    p = n_tp / n_pred if n_pred else 0.0
    r = n_tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def doc_counts(gold_facts, pred_facts, doc_ids):
    """Per-document (hits, predictions, gold) counts, as three arrays aligned with `doc_ids`."""
    gold = {f for f in gold_facts if f[2] != NO_RELATION}
    pred = {f for f in pred_facts if f[2] != NO_RELATION}
    index = {d: i for i, d in enumerate(doc_ids)}
    counts = np.zeros((3, len(doc_ids)))
    for row, facts in enumerate((gold & pred, pred, gold)):
        for f in facts:
            counts[row, index[f[0]]] += 1
    return counts


def _f1_from_counts(counts):
    """Micro F1 from (hits, predictions, gold) totals; works on arrays of totals."""
    tp, n_pred, n_gold = counts
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(n_pred + n_gold > 0, 2 * tp / (n_pred + n_gold), 0.0)
    return f1


def bootstrap_f1(counts, n_samples=1000, seed=0):
    """95% confidence interval of micro F1, resampling documents with replacement."""
    n_docs = counts.shape[1]
    draws = np.random.default_rng(seed).integers(0, n_docs, size=(n_samples, n_docs))
    f1 = _f1_from_counts(counts[:, draws].sum(axis=2))
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))


def paired_bootstrap(counts_a, counts_b, n_samples=1000, seed=0):
    """F1(a) - F1(b), its 95% interval and a one-sided p-value, on the same resampled documents.

    p is the share of resamples where a does not beat b: below 0.05, a is better beyond document sampling noise.
    """
    n_docs = counts_a.shape[1]
    draws = np.random.default_rng(seed).integers(0, n_docs, size=(n_samples, n_docs))
    delta = _f1_from_counts(counts_a[:, draws].sum(axis=2)) - _f1_from_counts(counts_b[:, draws].sum(axis=2))
    observed = float(_f1_from_counts(counts_a.sum(axis=1)) - _f1_from_counts(counts_b.sum(axis=1)))
    return observed, float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5)), float(np.mean(delta <= 0))


def ign_scores(gold_facts, pred_facts, in_train):
    """Ign precision and F1, official DocRED formula.

    Correct predictions that `in_train(fact)` marks as already seen in train (same head name, tail name
    and relation) are removed from the hits and from the predictions; recall is left unchanged.
    """
    gold = {f for f in gold_facts if f[2] != NO_RELATION}
    pred = {f for f in pred_facts if f[2] != NO_RELATION}
    tp = gold & pred
    seen = sum(1 for f in tp if in_train(f))
    p = (len(tp) - seen) / (len(pred) - seen) if len(pred) > seen else 0.0
    r = len(tp) / len(gold) if gold else 0.0
    return p, (2 * p * r / (p + r) if p + r else 0.0)


def typing_scores(gold_facts, pred_facts):
    """P/R/F1 on the pairs that are both related in gold and predicted related: relation choice only.

    Missed pairs and false pairs are detection errors and do not count here.
    """
    gold = {f for f in gold_facts if f[2] != NO_RELATION}
    pred = {f for f in pred_facts if f[2] != NO_RELATION}
    pairs = {(d, h, t) for d, h, _, t in gold} & {(d, h, t) for d, h, _, t in pred}
    gold = {f for f in gold if (f[0], f[1], f[3]) in pairs}
    pred = {f for f in pred if (f[0], f[1], f[3]) in pairs}
    return prf(len(gold & pred), len(pred), len(gold))
