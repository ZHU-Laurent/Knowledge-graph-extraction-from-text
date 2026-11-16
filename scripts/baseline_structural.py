"""M0': a structural-prior baseline with a detector, a typer and calibrated scores.

The type-pair baseline (M0, scripts/baseline_typepair.py) estimates P(relation | head type,
tail type, same sentence) by counting on train, and predicts every relation of a type pair whose
probability passes one global threshold. It has three limits:

- it cannot say "no relation". Every type pair that carries a relation in train contributes its
  best relations to every pair of that type, so precision stays near the base rate;
- the threshold is the same for the 96 relations, so a relation with a low prior can never be
  predicted, however related the pair is;
- it writes no score, so the graph metrics that need a confidence (precision at N, precision by
  number of supporting documents) cannot be computed for it at all.

This method keeps the promise of M0 -- it reads no text, only the structural metadata that the
dataset gives for free (entity types, mentions, sentence distance, Wikidata links) -- and fixes
the three limits. It has the same shape as PromptORE's M2 (relations/probe.py: a logistic
detector, then one logistic classifier per relation, and a fact score of
P(related) x P(relation | related)), with the structural features in place of the [MASK]
embeddings. The two therefore differ in exactly one thing, the representation of a pair, which
makes the comparison a clean answer to a question the leaderboard cannot answer yet: how much of
PromptORE's score comes from reading the text, and how much from the structural prior that the
type-pair baseline already has?

    python scripts/baseline_structural.py --run m0-structural
    python scripts/evaluate.py results/m0-structural
    python scripts/compare_runs.py results/m0-structural results/baseline-typepair --subset dev

Every choice is made on dev. The threshold rule is compared by cross-fitting on the dev
documents, so that the number reported for the run is not the number it was tuned on.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kg_extraction.data.linked_docred import load_relation_names, load_split  # noqa: E402
from kg_extraction.data.redocred import RedocredGold  # noqa: E402
from kg_extraction.evaluation.relations import gold_facts  # noqa: E402
from kg_extraction.evaluation.runs import write_shared_run  # noqa: E402
from kg_extraction.relations.probe import RelationProbe  # noqa: E402

BENCH = Path("data/processed/benchmark")
RAW = "data/raw/Linked-Re-DocRED"

# The buckets of the two structural features that are not one-hot types.
DISTANCE_BOUNDS = (1, 2, 3, 4, 6, 9)   # 7 buckets: 0, 1, 2, 3, 4-5, 6-8, 9+
MENTION_BOUNDS = (1, 2, 3, 4)          # 5 buckets: 1, 2, 3, 4, 5+


# --------------------------------------------------------------------------------------
# The structural description of the entities, and the features of the ordered pairs.
# No text is read: only the metadata of the dataset.
# --------------------------------------------------------------------------------------

def entity_info(document, types):
    """Return the structural description of every entity of a document."""
    info = []
    for entity in document["entities"]:
        sentences = sorted({mention["sent_id"] for mention in entity["mentions"]})
        info.append({
            "type": types.index(entity["type"]),
            "sentences": sentences,
            "first_sentence": sentences[0] if sentences else -1,
            "mentions": len(entity["mentions"]),
            "wikidata": 1 if entity.get("wikidata") else 0,
        })
    return info


def sentence_distance(head, tail):
    """The smallest distance between a sentence of the head and a sentence of the tail."""
    return min(abs(first - second) for first in head["sentences"] for second in tail["sentences"])


def feature_width(n_types, version):
    """The number of features of one pair.

    v0 holds exactly the information M0 uses (the joint type-pair key), so that the gain of
    the model can be told apart from the gain of the extra information. v1 adds the other
    structural features, v2 adds the interactions between the type pair and the distance.
    """
    width = (2 * n_types          # head type, tail type
             + 1                  # same sentence
             + n_types ** 2       # type pair
             + n_types ** 2)      # type pair x same sentence
    if version in ("v1", "v2"):
        width += len(DISTANCE_BOUNDS) + 1
        width += 2 * (len(MENTION_BOUNDS) + 1)
        width += 2                  # Wikidata links
        width += 1                  # same type
        width += 1                  # size of the document
    if version == "v2":
        width += n_types ** 2 * (len(DISTANCE_BOUNDS) + 1)   # type pair x sentence distance
        width += 3                                          # subject position, first mention
    return width


def pair_features(info, n_types, version):
    """Return the feature matrix of every ordered pair of a document, and the pair list.

    The pairs are in the order of the nested loops over the entities, which is the order of
    the pairs file of the benchmark, so the two can be joined by position.
    """
    width = feature_width(n_types, version)
    n = len(info)
    heads = np.repeat(np.arange(n), n)
    tails = np.tile(np.arange(n), n)
    keep = heads != tails
    heads, tails = heads[keep], tails[keep]
    m = len(heads)
    rows = np.arange(m)

    features = np.zeros((m, width), dtype=np.float32)
    head_type = np.array([info[index]["type"] for index in heads])
    tail_type = np.array([info[index]["type"] for index in tails])
    shared = np.array([1.0 if set(info[h]["sentences"]) & set(info[t]["sentences"]) else 0.0
                        for h, t in zip(heads, tails)], dtype=np.float32)

    features[rows, head_type] = 1.0
    features[rows, n_types + tail_type] = 1.0
    features[rows, 2 * n_types] = shared
    pair_index = head_type * n_types + tail_type
    features[rows, 2 * n_types + 1 + pair_index] = 1.0
    features[rows, 2 * n_types + 1 + n_types ** 2 + pair_index] = shared

    offset = 2 * n_types + 1 + 2 * n_types ** 2
    distance_bucket = None
    if version in ("v1", "v2"):
        distance_bucket = np.empty(m, dtype=np.int64)
        for row, (head, tail) in enumerate(zip(heads, tails)):
            distance = sentence_distance(info[head], info[tail])
            bucket = int(np.digitize(distance, DISTANCE_BOUNDS))
            distance_bucket[row] = bucket
            features[row, offset + bucket] = 1.0
        distance_offset = offset
        offset += len(DISTANCE_BOUNDS) + 1

        for side in (heads, tails):
            counts = np.array([info[index]["mentions"] for index in side])
            buckets = np.digitize(counts, MENTION_BOUNDS)
            features[rows, offset + buckets] = 1.0
            offset += len(MENTION_BOUNDS) + 1

        features[rows, offset] = np.array([info[index]["wikidata"] for index in heads])
        features[rows, offset + 1] = np.array([info[index]["wikidata"] for index in tails])
        offset += 2
        features[rows, offset] = (head_type == tail_type).astype(np.float32)
        features[rows, offset + 1] = np.log1p(n)
        offset += 2

    if version == "v2":
        # The type pair and the sentence distance together: how far apart two entities of
        # these types are says more than either of them alone.
        base = offset + n_types ** 2 * distance_bucket + pair_index
        features[rows, base] = 1.0
        offset += n_types ** 2 * (len(DISTANCE_BOUNDS) + 1)
        # The subject of a Wikipedia introduction is its first entity, and the sentences of
        # the first mention of each entity are where its properties are stated.
        features[rows, offset] = (heads == 0).astype(np.float32)
        features[rows, offset + 1] = (tails == 0).astype(np.float32)
        features[rows, offset + 2] = np.array(
            [1.0 if info[h]["first_sentence"] == info[t]["first_sentence"] else 0.0
             for h, t in zip(heads, tails)], dtype=np.float32)

    pairs = list(zip(heads.tolist(), tails.tolist()))
    return features, pairs


def subset_matrix(subset, types, version):
    """Return the feature matrix of every ordered pair of a subset, and the pair list.

    The pairs are (document index, head id, tail id), the document index being the position of
    the document in the split, as in relations/probe.py:pair_list. The matrix is filled in
    place, because the train matrix holds more than a hundred million cells.
    """
    infos, sizes = [], []
    with open(BENCH / f"redocred_{subset}_docs.jsonl", encoding="utf-8") as file:
        for line in file:
            document = json.loads(line)
            infos.append(entity_info(document, types))
            sizes.append(len(document["entities"]))
    width = feature_width(len(types), version)
    matrix = np.empty((sum(size * (size - 1) for size in sizes), width), dtype=np.float32)

    pairs, offset = [], 0
    for index, info in enumerate(infos):
        features, local = pair_features(info, len(types), version)
        matrix[offset:offset + len(local)] = features
        pairs.extend((index, head, tail) for head, tail in local)
        offset += len(local)
    return matrix, pairs


def labels_for(pairs, gold):
    """Return the gold relation codes of every pair, in the order of the pair list."""
    return [sorted(gold.pair_labels.get((gold.doc_ids[index], head, tail), ()))
            for index, head, tail in pairs]


# --------------------------------------------------------------------------------------
# The fact threshold, chosen on dev. One global threshold, or one per relation.
# --------------------------------------------------------------------------------------

def correctness_matrix(scores, pairs, relations, gold):
    """Return a boolean matrix: is the fact of this pair row and relation column in the gold?"""
    pair_index = {(index, head, tail): row for row, (index, head, tail) in enumerate(pairs)}
    column = {code: number for number, code in enumerate(relations)}
    correct = np.zeros(scores.shape, dtype=bool)
    for fact in gold:
        row = pair_index.get((fact[0], fact[1], fact[2]))
        if row is not None:
            correct[row, column[fact[3]]] = True
    return correct


def best_global_threshold(scores, correct, rows, n_gold):
    """The threshold that maximises the micro F1 over the given pair rows."""
    values = scores[rows].ravel()
    hits = correct[rows].ravel()
    order = np.argsort(-values, kind="stable")
    values, hits = values[order], hits[order]
    cumulative = np.cumsum(hits)
    ranks = np.arange(1, values.size + 1)
    f1 = 2 * cumulative / (ranks + n_gold)
    best = int(np.argmax(f1))
    return float(values[best]), float(100 * f1[best])


def best_relation_thresholds(scores, correct, rows):
    """One threshold per relation, each maximising the F1 of that relation."""
    thresholds = np.zeros(scores.shape[1], dtype=np.float32)
    for column in range(scores.shape[1]):
        values = scores[rows, column]
        hits = correct[rows, column]
        n_gold = int(hits.sum())
        order = np.argsort(-values, kind="stable")
        values, hits = values[order], hits[order]
        cumulative = np.cumsum(hits)
        ranks = np.arange(1, values.size + 1)
        f1 = 2 * cumulative / (ranks + n_gold)
        thresholds[column] = values[int(np.argmax(f1))]
    return thresholds


def micro_f1_counts(scores, correct, rows, thresholds):
    """Return the (hits, predicted, gold) counts of the facts kept by the thresholds."""
    kept = scores[rows] >= (thresholds if np.isscalar(thresholds) else thresholds[None, :])
    return int((kept & correct[rows]).sum()), int(kept.sum()), int(correct[rows].sum())


def micro_f1(scores, correct, rows, thresholds):
    """The micro F1 of the facts kept by the thresholds."""
    hits, predicted, gold = micro_f1_counts(scores, correct, rows, thresholds)
    return 2 * hits / (predicted + gold) if predicted + gold else 0.0


def apply_thresholds(scores, pairs, relations, thresholds):
    """Return {(doc index, head, tail, relation): score} for the facts above the thresholds."""
    if np.isscalar(thresholds):
        rows, columns = np.nonzero(scores >= thresholds)
    else:
        rows, columns = np.nonzero(scores >= thresholds[None, :])
    return {(pairs[row][0], pairs[row][1], pairs[row][2], relations[column]): float(scores[row, column])
            for row, column in zip(rows.tolist(), columns.tolist())}


def cross_fitted_f1(scores, correct, pairs, relations, rule, n_folds=5, seed=0):
    """The dev F1 of a threshold rule, tuned on n-1 folds and scored on the held-out fold.

    The folds are documents, so a fact is never tuned and scored on the same document. The
    number returned is the F1 of the pooled out-of-fold facts, so that it can be compared
    with the F1 of the whole set: the mean of the F1 of the folds would be biased upwards
    when the folds differ a lot, because it is a mean of ratios instead of a ratio of sums.
    """
    doc_of_pair = np.array([index for index, _, _ in pairs])
    documents = np.unique(doc_of_pair)
    fold_of_doc = {int(doc): number for number, doc in
                   enumerate(np.random.default_rng(seed).permutation(documents))}
    fold_of_pair = np.array([fold_of_doc[int(doc)] for doc in doc_of_pair])
    per_fold, totals = [], [0, 0, 0]
    for fold in range(n_folds):
        train_rows = np.nonzero(fold_of_pair != fold)[0]
        test_rows = np.nonzero(fold_of_pair == fold)[0]
        if rule == "global":
            threshold, _ = best_global_threshold(scores, correct, train_rows,
                                                 int(correct[train_rows].sum()))
            thresholds = threshold
        else:
            thresholds = best_relation_thresholds(scores, correct, train_rows)
        hits, predicted, gold = micro_f1_counts(scores, correct, test_rows, thresholds)
        per_fold.append(2 * hits / (predicted + gold) if predicted + gold else 0.0)
        totals = [totals[0] + hits, totals[1] + predicted, totals[2] + gold]
    pooled = 2 * totals[0] / (totals[1] + totals[2]) if totals[1] + totals[2] else 0.0
    return float(100 * pooled), per_fold


# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="m0-structural")
    parser.add_argument("--author", default="Bai-Xiaoyi")
    parser.add_argument("--data-dir", default=RAW)
    parser.add_argument("--runs-dir", default="results")
    parser.add_argument("--features", default="v2", choices=["v0", "v1", "v2"])
    parser.add_argument("--thresholds", default="per-relation", choices=["global", "per-relation"])
    parser.add_argument("--negatives-per-positive", type=int, default=3,
                        help="unrelated train pairs per related one; 0 uses all of them")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true",
                        help="also predict test; only after every choice is made on dev")
    args = parser.parse_args()

    started = time.time()
    relations = sorted(load_relation_names(args.data_dir))
    train_documents = load_split(args.data_dir, "train")
    types = sorted({entity["type"] for document in train_documents for entity in document["entities"]})
    print(f"{len(relations)} relations, {len(types)} entity types {types}, features {args.features}",
          flush=True)

    print("building the train features...", flush=True)
    train_features, train_pairs = subset_matrix("train", types, args.features)
    print(f"  {len(train_pairs)} train pairs, {train_features.shape[1]} features", flush=True)

    gold_train = RedocredGold("train", args.data_dir)
    gold_train.doc_ids = list(gold_train.docs)
    train_labels = labels_for(train_pairs, gold_train)
    related = np.array([bool(codes) for codes in train_labels])
    print(f"  {related.sum()} related pairs", flush=True)

    if args.negatives_per_positive > 0:
        kept = [index for index, flag in enumerate(related) if flag]
        unrelated = [index for index, flag in enumerate(related) if not flag]
        n_sampled = min(len(unrelated), args.negatives_per_positive * len(kept))
        drawn = np.random.default_rng(args.seed).choice(unrelated, size=n_sampled, replace=False)
        selected = np.sort(np.array(kept + drawn.tolist()))
        negative_weight = len(unrelated) / max(n_sampled, 1)
    else:
        selected = np.arange(len(train_pairs))
        negative_weight = 1.0
    print(f"  training on {len(selected)} pairs, negative weight {negative_weight:.1f}", flush=True)

    probe = RelationProbe(relations).fit(train_features[selected], [train_labels[i] for i in selected],
                                         negative_weight)
    del train_features
    print(f"probe fitted in {time.time() - started:.0f}s", flush=True)

    print("scoring dev...", flush=True)
    dev_features, dev_pairs = subset_matrix("dev", types, args.features)
    dev_scores = probe.scores(dev_features).astype(np.float32)
    del dev_features
    dev_documents = load_split(args.data_dir, "dev")
    gold_dev = gold_facts(dev_documents)
    correct = correctness_matrix(dev_scores, dev_pairs, relations, gold_dev)
    all_rows = np.arange(len(dev_pairs))
    n_gold = int(correct.sum())

    if args.thresholds == "global":
        threshold, dev_f1 = best_global_threshold(dev_scores, correct, all_rows, n_gold)
        thresholds = threshold
        print(f"  dev F1 {dev_f1:.2f} at global threshold {threshold:.4f}", flush=True)
    else:
        thresholds = best_relation_thresholds(dev_scores, correct, all_rows)
        dev_f1 = 100 * micro_f1(dev_scores, correct, all_rows, thresholds)
        print(f"  dev F1 {dev_f1:.2f} with one threshold per relation", flush=True)

    for rule in ("global", "per-relation"):
        value, folds = cross_fitted_f1(dev_scores, correct, dev_pairs, relations, rule, seed=args.seed)
        print(f"  cross-fitted dev F1, {rule:12s}: {value:.2f}  (folds "
              + ", ".join(f"{100 * f:.1f}" for f in folds) + ")", flush=True)

    facts = {"dev": apply_thresholds(dev_scores, dev_pairs, relations, thresholds)}
    documents = {"dev": dev_documents}
    if args.test:
        print("scoring test...", flush=True)
        test_features, test_pairs = subset_matrix("test", types, args.features)
        test_scores = probe.scores(test_features).astype(np.float32)
        facts["test"] = apply_thresholds(test_scores, test_pairs, relations, thresholds)
        documents["test"] = load_split(args.data_dir, "test")

    description = ("no text read: entity types, mentions, sentence distance and Wikidata links; "
                   f"features {args.features}; "
                   f"{args.negatives_per_positive} unrelated train pairs per related one, seed {args.seed}; "
                   f"{args.thresholds} fact threshold chosen on dev; "
                   "fact score = P(related) x P(relation | related)")
    card = {
        "run": args.run,
        "author": args.author,
        "method": "Structural-prior baseline (M0'): logistic detector + per-relation logistic typer",
        "supervision": "supervised",
        "training_data": "Linked-Re-DocRED train labels",
        "models": "none (logistic regression on structural features)",
        "description": description,
    }
    write_shared_run(Path(args.runs_dir) / args.run, documents, facts, card, args.seed)
    print(f"wrote {Path(args.runs_dir) / args.run} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
