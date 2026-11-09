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
