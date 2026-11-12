# ===================== Linked-Re-DocRED shared evaluation (paste as ONE notebook cell) =====================
# Same subsets, same metrics as scripts/evaluate.py, so numbers are comparable across members. Guide: docs/evaluation.md
#
#   docs  = load_subset("dev_debug")        # list of documents: doc_id, title, sents, entities (NO relations)
#   pairs = candidate_pairs("dev_debug")    # DataFrame: doc_id, h, t, h_type, t_type, same_sentence
#   ... your method ...  -> predictions: iterable of (doc_id, h, r, t), or a DataFrame with those 4 columns
#   scores = evaluate_redocred(predictions, "dev_debug")                      # facts
#   scores = evaluate_redocred(predictions, "dev_debug", clusters=cl)          # + B3/V/ARI, cl: (doc_id, h, t, cluster)
#   gold_train = gold_facts("train")        # train labels, for methods allowed to use them
#
# Subsets (seed 0): train_debug = 300 train docs, dev_debug = 100 dev docs, train / dev / test = full splits.
# Rules: tune on dev / dev_debug, score test once at the end. "no_relation" rows are ignored.
import json, os, random, urllib.request
from collections import Counter
import numpy as np, pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score, completeness_score, contingency_matrix, homogeneity_score

REDOCRED_DIR = "data/raw/Linked-Re-DocRED"
_URL = "https://raw.githubusercontent.com/alteca/Linked-DocRED/main/Linked-Re-DocRED/{}_revised.json"
_SUBSETS = {"train_debug": ("train", 300), "dev_debug": ("dev", 100), "train": ("train", None),
            "dev": ("dev", None), "test": ("test", None)}
_SPLITS = {}


def _split(split):
    if split not in _SPLITS:
        path = os.path.join(REDOCRED_DIR, f"{split}_revised.json")
        if not os.path.exists(path):
            os.makedirs(REDOCRED_DIR, exist_ok=True)
            print("downloading", split, "...")
            urllib.request.urlretrieve(_URL.format(split), path)
        with open(path, encoding="utf-8") as f:
            _SPLITS[split] = json.load(f)
    return _SPLITS[split]


def _subset_docs(subset):
    split, size = _SUBSETS[subset]
    docs = _split(split)
    ids = sorted(str(d["id"]) for d in docs)
    if size is not None and size < len(ids):
        ids = sorted(random.Random(0).sample(ids, size))
    keep = set(ids)
    return [d for d in docs if str(d["id"]) in keep]


def _same_sentence(doc, h, t):
    return bool({m["sent_id"] for m in doc["entities"][h]["mentions"]} &
                {m["sent_id"] for m in doc["entities"][t]["mentions"]})


def load_subset(subset):
    """Documents without relations. Entity `name` = longest mention."""
    return [{"doc_id": str(d["id"]), "title": d["title"], "sents": d["sents"],
             "entities": [{"id": i, "type": e["type"], "name": max((m["name"] for m in e["mentions"]), key=len),
                           "mentions": e["mentions"]} for i, e in enumerate(d["entities"])]}
            for d in _subset_docs(subset)]


def candidate_pairs(subset):
    rows = []
    for d in _subset_docs(subset):
        E = d["entities"]
        for h in range(len(E)):
            for t in range(len(E)):
                if h != t:
                    rows.append((str(d["id"]), h, t, E[h]["type"], E[t]["type"], int(_same_sentence(d, h, t))))
    return pd.DataFrame(rows, columns=["doc_id", "h", "t", "h_type", "t_type", "same_sentence"])


def gold_facts(subset, with_evidence=False):
    """{(doc_id, h, r, t)}; with_evidence=True keeps only facts annotated with evidence sentences."""
    return {(str(d["id"]), int(r["h"]), r["r"], int(r["t"])) for d in _subset_docs(subset)
            for r in d["relations"] if r.get("evidence") or not with_evidence}


def _prf(tp, n_pred, n_gold):
    p, r = (tp / n_pred if n_pred else 0.0), (tp / n_gold if n_gold else 0.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _facts(gold, pred):
    tp = gold & pred
    ng, npr, nt = Counter(f[2] for f in gold), Counter(f[2] for f in pred), Counter(f[2] for f in tp)
    per = {r: dict(zip(("p", "r", "f1"), _prf(nt[r], npr[r], ng[r])), gold=ng[r], pred=npr[r], tp=nt[r]) for r in ng}
    p, r, f1 = _prf(len(tp), len(pred), len(gold))
    return p, r, f1, per


def clustering_scores(gold, pred):
    c = contingency_matrix(gold, pred)
    w = c / c.sum()
    b3p, b3r = float(np.sum(w * (c / c.sum(axis=0)))), float(np.sum(w * (c / c.sum(axis=1, keepdims=True))))
    hom, comp = homogeneity_score(gold, pred), completeness_score(gold, pred)
    return {"b3_f1": 2 * b3p * b3r / (b3p + b3r) if b3p + b3r else 0.0, "b3_p": b3p, "b3_r": b3r,
            "v_f1": 2 * hom * comp / (hom + comp) if hom + comp else 0.0,
            "ari": adjusted_rand_score(gold, pred), "n_clusters": len(set(pred))}


def evaluate_redocred(predictions, subset="dev_debug", clusters=None, verbose=True):
    docs = {str(d["id"]): d for d in _subset_docs(subset)}
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions[["doc_id", "h", "r", "t"]].itertuples(index=False, name=None)
    pred = set()
    for doc, h, r, t in predictions:
        doc, h, t = str(doc), int(h), int(t)
        assert doc in docs, f"document {doc} is not in subset {subset}"
        assert h != t and 0 <= h < len(docs[doc]["entities"]) and 0 <= t < len(docs[doc]["entities"]), (doc, h, t)
        if r != "no_relation":
            pred.add((doc, h, r, t))

    gold, gold_ev = gold_facts(subset), gold_facts(subset, with_evidence=True)
    p, r, f1, per = _facts(gold, pred)
    no_ev = gold - gold_ev                                   # neither hits nor errors in the evidence view
    ep, er, ef1, _ = _facts(gold_ev, pred - no_ev)
    s = {"micro_p": p, "micro_r": r, "micro_f1": f1, "macro_f1": float(np.mean([v["f1"] for v in per.values()])),
         "evid_p": ep, "evid_r": er, "evid_f1": ef1, "n_gold": len(gold), "n_pred": len(pred), "n_tp": len(gold & pred)}

    for name, keep in (("intra", True), ("inter", False)):
        g = [f for f in gold if _same_sentence(docs[f[0]], f[1], f[3]) == keep]
        pk = [f for f in pred if _same_sentence(docs[f[0]], f[1], f[3]) == keep]
        s[f"recall_{name}_sentence"] = sum(f in pred for f in g) / len(g) if g else 0.0
        s[f"f1_{name}_sentence"] = _prf(sum(f in pred for f in g), len(pk), len(g))[2]

    # Ign F1 (DocRED): correct facts whose mention names + relation already form a train fact are not counted
    names = {(mh["name"], mt["name"], x["r"]) for d in _split("train") for x in d["relations"]
             for mh in d["entities"][x["h"]]["mentions"] for mt in d["entities"][x["t"]]["mentions"]}
    seen = sum(1 for d0, h0, r0, t0 in gold & pred
               if any((mh["name"], mt["name"], r0) in names for mh in docs[d0]["entities"][h0]["mentions"]
                      for mt in docs[d0]["entities"][t0]["mentions"]))
    s["ign_p"] = (len(gold & pred) - seen) / (len(pred) - seen) if len(pred) > seen else 0.0
    s["ign_f1"] = 2 * s["ign_p"] * r / (s["ign_p"] + r) if s["ign_p"] + r else 0.0

    # Typing: only pairs related in gold and predicted related; is the relation right?
    both = {(d0, h0, t0) for d0, h0, _, t0 in gold} & {(d0, h0, t0) for d0, h0, _, t0 in pred}
    tg, tpred = {f for f in gold if (f[0], f[1], f[3]) in both}, {f for f in pred if (f[0], f[1], f[3]) in both}
    s["typing_p"], s["typing_r"], s["typing_f1"] = _prf(len(tg & tpred), len(tpred), len(tg))

    counts = Counter(x["r"] for d in _split("train") for x in d["relations"])
    rank = {rel: i for i, (rel, _) in enumerate(counts.most_common())}
    bucket = lambda rel: "frequent" if rank.get(rel, 999) < 10 else ("rare" if counts[rel] < 200 else "medium")
    for b in ("frequent", "medium", "rare"):
        f1s = [v["f1"] for rel, v in per.items() if bucket(rel) == b]
        s[f"macro_f1_{b}"] = float(np.mean(f1s)) if f1s else 0.0

    labels = {}
    for d0, h0, r0, t0 in gold:
        labels.setdefault((d0, h0, t0), set()).add(r0)
    pp = {(d0, h0, t0) for d0, h0, _, t0 in pred}
    s["detect_p"], s["detect_r"], s["detect_f1"] = _prf(len(pp & set(labels)), len(pp), len(labels))

    # 95% interval of micro F1, resampling documents (same draws as scripts/evaluate.py)
    ids = sorted(docs)
    pos = {d0: i for i, d0 in enumerate(ids)}
    counts = np.zeros((3, len(ids)))
    for row, fs in enumerate((gold & pred, pred, gold)):
        for f in fs:
            counts[row, pos[f[0]]] += 1
    draws = np.random.default_rng(0).integers(0, len(ids), size=(1000, len(ids)))
    tp_, np_, ng_ = counts[:, draws].sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.where(np_ + ng_ > 0, 2 * tp_ / (np_ + ng_), 0.0)
    s["micro_f1_ci_low"], s["micro_f1_ci_high"] = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    if clusters is not None:
        if isinstance(clusters, pd.DataFrame):
            clusters = clusters[["doc_id", "h", "t", "cluster"]].itertuples(index=False, name=None)
        cl = {(str(d0), int(h0), int(t0)): c for d0, h0, t0, c in clusters}
        keys = [k for k, v in labels.items() if len(v) == 1 and k in cl]
        if keys:
            s.update({f"cl_{k}": v for k, v in clustering_scores([next(iter(labels[k])) for k in keys],
                                                                  [cl[k] for k in keys]).items()})
    if verbose:
        print(f"[{subset}] KEY  F1 = {f1:.3f} (95% interval {s['micro_f1_ci_low']:.3f} to {s['micro_f1_ci_high']:.3f})"
              f"   Ign F1 = {s['ign_f1']:.3f}")
        print(f"          P/R = {p:.3f} / {r:.3f}   detection F1 = {s['detect_f1']:.3f}   "
              f"typing F1 = {s['typing_f1']:.3f}   macro F1 = {s['macro_f1']:.3f}   evidence-only F1 = {ef1:.3f}")
        print(f"          recall same sentence = {s['recall_intra_sentence']:.3f}, cross sentence = "
              f"{s['recall_inter_sentence']:.3f}   macro F1 frequent/medium/rare = {s['macro_f1_frequent']:.3f}/"
              f"{s['macro_f1_medium']:.3f}/{s['macro_f1_rare']:.3f}")
        if "cl_b3_f1" in s:
            print(f"          clusters: B3 F1 = {s['cl_b3_f1']:.3f}  V F1 = {s['cl_v_f1']:.3f}  ARI = {s['cl_ari']:.3f}")
    s["per_relation"] = pd.DataFrame(per).T.sort_values("gold", ascending=False)
    return s
# ============================================================================================================
