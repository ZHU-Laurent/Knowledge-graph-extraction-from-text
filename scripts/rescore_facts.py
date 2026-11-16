"""Rescore the facts of a run with rules mined from train and a model fitted on dev.

The run's scores are kept down to --min-score (the saved scores go down to 0.02), so
facts under the run's threshold can come back. Every variant is scored on dev only:

- base: the run's scores, threshold chosen on dev (what the leaderboard shows);
- rules: hard rules mined from train, then a new threshold: relation / entity-type
  combinations never seen in train are dropped, and a functional relation (one tail
  per head in train) keeps its best tail only;
- buckets: one threshold per relation frequency bucket (rare, medium, frequent);
- rescore-<features>: a gradient boosting model on features of each candidate fact,
  fitted on 4 folds of dev documents and applied to the fifth, so every dev fact gets
  a score from a model that did not see its document. Feature groups:
    scores     the run's score and its members' scores;
    pair       the other relations of the same pair, the rank of the tail among the
               head's tails for this relation, and of the head among the tail's heads;
    inverse    the score of the reverse fact with the relation that most often comes
               back reversed in train (the same one for a symmetric relation);
    types      how often the relation joins these entity types in train, how often a
               head has several tails (functional relations);
    chain      the best two-step path h -a-> z -b-> t whose rule (a, b) => r holds in train;
    doc        sentence distance, shared sentence, mentions, entities of the document;
    corpus     how many other documents of the split predict the same Wikidata fact;
    prior      whether the Wikidata fact is in the train graph (memorised facts, so
               compare it on Ign F1).

  rescore_facts.py --run results/doc-tune-ens+old:0 \\
      --members results/doc-tune-ens:0 results/doc-ens+m2:0
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from kg_extraction.data.linked_docred import load_split
from kg_extraction.evaluation.relations import gold_facts, relation_scores, train_fact_names
from kg_extraction.relations.doc_model import best_threshold

TYPES = ["PER", "ORG", "LOC", "TIME", "NUM", "MISC"]
GROUPS = ["scores", "pair", "inverse", "types", "chain", "doc", "corpus", "prior"]


def load_scores(member, split):
    run_dir, seed = member.rsplit(":", 1)
    return torch.load(Path(run_dir) / f"scores_{split}_seed{seed}.pt", weights_only=False)


def wikidata_id(entity):
    """Return the Wikidata item of an entity, or None for dates, numbers and unlinked entities."""
    link = (entity.get("entity_linking") or {}).get("wikidata_resource") or ""
    return link if link.startswith("Q") else None


def sentences(entity):
    return {mention["sent_id"] for mention in entity["mentions"]}


class TrainStats:
    """Counts of the train facts that the rules and features use."""

    def __init__(self, train, min_rule_support=20, min_rule_confidence=0.2):
        self.relation_count, self.type_count = Counter(), Counter()
        tails, heads = defaultdict(set), defaultdict(set)
        reverse, self.graph = Counter(), set()
        paths, closed = Counter(), Counter()
        for index, doc in enumerate(train):
            entities = doc["entities"]
            facts = {(r["h"], r["t"], r["r"]) for r in doc["relations"]}
            out = defaultdict(list)
            for head, tail, relation in facts:
                self.relation_count[relation] += 1
                self.type_count[relation, entities[head]["type"], entities[tail]["type"]] += 1
                tails[index, head, relation].add(tail)
                heads[index, tail, relation].add(head)
                out[head].append((relation, tail))
                ids = wikidata_id(entities[head]), wikidata_id(entities[tail])
                if all(ids):
                    self.graph.add((ids[0], relation, ids[1]))
            pair_relations = defaultdict(set)
            for head, tail, relation in facts:
                pair_relations[head, tail].add(relation)
            for head, tail, relation in facts:
                for back in pair_relations.get((tail, head), ()):
                    reverse[relation, back] += 1
            for head, steps in out.items():
                for first, middle in steps:
                    for second, tail in out.get(middle, ()):
                        if tail == head:
                            continue
                        paths[first, second] += 1
                        for relation in pair_relations.get((head, tail), ()):
                            closed[first, second, relation] += 1

        self.several_tails = self._share_several(tails)
        self.several_heads = self._share_several(heads)
        self.inverse = {}
        for (relation, back), count in reverse.items():
            confidence = count / self.relation_count[relation]
            if confidence > self.inverse.get(relation, (None, 0.0))[1]:
                self.inverse[relation] = (back, confidence)
        self.rules = defaultdict(list)
        for (first, second, relation), count in closed.items():
            confidence = count / paths[first, second]
            if count >= min_rule_support and confidence >= min_rule_confidence:
                self.rules[relation].append((first, second, confidence))

    @staticmethod
    def _share_several(groups):
        total, several = Counter(), Counter()
        for (_, _, relation), members in groups.items():
            total[relation] += 1
            several[relation] += len(members) > 1
        return {relation: several[relation] / total[relation] for relation in total}

    def functional(self, relation, limit=0.05):
        return self.several_tails.get(relation, 1.0) < limit


def candidates(scores, min_score):
    return {fact: score for fact, score in scores.items() if score >= min_score}


def features(docs, facts, members, stats, groups):
    """Return the feature matrix of the facts (a list) and the indices of categorical columns."""
    relations = sorted(stats.relation_count)
    relation_index = {relation: i for i, relation in enumerate(relations)}
    scores = dict(facts)
    facts = [fact for fact, _ in facts]
    by_pair, by_head, by_tail, by_step = (defaultdict(list) for _ in range(4))
    for (index, head, tail, relation), score in scores.items():
        by_pair[index, head, tail].append(score)
        by_head[index, head, relation].append(score)
        by_tail[index, tail, relation].append(score)
        by_step[index, head, relation].append((tail, score))
    corpus = defaultdict(list)
    for (index, head, tail, relation), score in scores.items():
        entities = docs[index]["entities"]
        ids = wikidata_id(entities[head]), wikidata_id(entities[tail])
        if all(ids):
            corpus[ids[0], relation, ids[1]].append((index, score))

    def rank(values, score):
        return sum(value > score for value in values)

    rows, categorical = [], []
    for fact in facts:
        index, head, tail, relation = fact
        score = scores[fact]
        entities = docs[index]["entities"]
        first, second = entities[head], entities[tail]
        row, names = [], []
        if "scores" in groups:
            row += [score] + [member.get(fact, 0.0) for member in members]
            row.append(relation_index.get(relation, len(relations)))
            names += ["score"] + [f"member{i}" for i in range(len(members))] + ["relation"]
        if "pair" in groups:
            pair = by_pair[index, head, tail]
            row += [max([s for s in pair if s != score], default=0.0), len(pair), rank(pair, score),
                    rank(by_head[index, head, relation], score), len(by_head[index, head, relation]),
                    rank(by_tail[index, tail, relation], score), len(by_tail[index, tail, relation])]
            names += ["pair_other", "pair_n", "pair_rank", "head_rank", "head_n", "tail_rank", "tail_n"]
        if "inverse" in groups:
            back, confidence = stats.inverse.get(relation, (relation, 0.0))
            row += [scores.get((index, tail, head, back), 0.0), confidence,
                    scores.get((index, tail, head, relation), 0.0)]
            names += ["inverse", "inverse_confidence", "symmetric"]
        if "types" in groups:
            seen = stats.type_count[relation, first["type"], second["type"]]
            row += [np.log1p(seen), np.log1p(stats.relation_count[relation]),
                    stats.several_tails.get(relation, 1.0), stats.several_heads.get(relation, 1.0),
                    TYPES.index(first["type"]), TYPES.index(second["type"])]
            names += ["type_seen", "relation_count", "several_tails", "several_heads", "head_type", "tail_type"]
        if "chain" in groups:
            best = 0.0
            for a, b, confidence in stats.rules.get(relation, ()):
                for middle, step in by_step[index, head, a]:
                    if middle != tail:
                        best = max(best, confidence * min(step, scores.get((index, middle, tail, b), 0.0)))
            row.append(best)
            names.append("chain")
        if "doc" in groups:
            a, b = sentences(first), sentences(second)
            row += [min(abs(x - y) for x in a for y in b), bool(a & b), len(first["mentions"]),
                    len(second["mentions"]), len(entities)]
            names += ["distance", "same_sentence", "head_mentions", "tail_mentions", "entities"]
        if "corpus" in groups or "prior" in groups:
            ids = wikidata_id(first), wikidata_id(second)
            key = (ids[0], relation, ids[1]) if all(ids) else None
            if "corpus" in groups:
                others = [s for i, s in corpus.get(key, ()) if i != index] if key else []
                row += [len(others), sum(s >= 0.5 for s in others), max(others, default=0.0)]
                names += ["corpus_n", "corpus_confident", "corpus_max"]
            if "prior" in groups:
                row.append(key in stats.graph if key else -1)
                names.append("in_train_graph")
        rows.append(row)
    categorical = [i for i, name in enumerate(names) if name in ("relation", "head_type", "tail_type")]
    return np.array(rows, dtype=float), categorical, names


def folds_of(docs, n_folds, seed=0):
    order = np.random.default_rng(seed).permutation(len(docs))
    fold = np.empty(len(docs), dtype=int)
    fold[order] = np.arange(len(docs)) % n_folds
    return fold


def cross_fitted(docs, facts, matrix, categorical, gold, n_folds):
    """Return out-of-fold probabilities of the facts: each fold is scored by a model fitted on the others."""
    fold = folds_of(docs, n_folds)[[fact[0] for fact in facts]]
    labels = np.array([fact in gold for fact in facts])
    probabilities = np.zeros(len(facts))
    for k in range(n_folds):
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, categorical_features=categorical,
                                               random_state=0)
        model.fit(matrix[fold != k], labels[fold != k])
        probabilities[fold == k] = model.predict_proba(matrix[fold == k])[:, 1]
    return dict(zip(facts, probabilities))


def threshold_cross_fitted(docs, scores, gold, n_folds):
    """F1 when the threshold of each fold is chosen on the other folds (no threshold tuned on its own documents)."""
    fold = folds_of(docs, n_folds, seed=1)
    kept = set()
    for k in range(n_folds):
        inside = {f: s for f, s in scores.items() if fold[f[0]] != k}
        threshold, _ = best_threshold(inside, {f for f in gold if fold[f[0]] != k})
        kept |= {f for f, s in scores.items() if fold[f[0]] == k and s >= threshold}
    return round(200 * len(kept & gold) / (len(kept) + len(gold)), 2)


def apply_rules(docs, scores, stats):
    kept = {}
    for fact, score in scores.items():
        index, head, tail, relation = fact
        entities = docs[index]["entities"]
        if stats.type_count[relation, entities[head]["type"], entities[tail]["type"]]:
            kept[fact] = score
    best = {}
    for fact, score in kept.items():
        index, head, tail, relation = fact
        if stats.functional(relation):
            key = (index, head, relation)
            if score > best.get(key, (None, -1))[1]:
                best[key] = (fact, score)
    return {fact: score for fact, score in kept.items()
            if not stats.functional(fact[3]) or best[fact[0], fact[1], fact[3]][0] == fact}


def bucket_scores(docs, scores, gold, stats, n_folds):
    """F1 with one threshold per relation frequency bucket, each chosen on the other folds."""
    def bucket(relation):
        count = stats.relation_count[relation]
        return 0 if count < 200 else 1 if count < 1000 else 2
    fold = folds_of(docs, n_folds, seed=1)
    kept = set()
    for k in range(n_folds):
        for b in range(3):
            inside = {f: s for f, s in scores.items() if fold[f[0]] != k and bucket(f[3]) == b}
            threshold, _ = best_threshold(inside, {f for f in gold if fold[f[0]] != k and bucket(f[3]) == b})
            kept |= {f for f, s in scores.items() if fold[f[0]] == k and bucket(f[3]) == b and s >= threshold}
    return kept


def report(name, docs, kept, names_in_train, extra=None):
    scores = relation_scores(docs, kept, names_in_train)
    row = {"variant": name, "f1": scores["f1"], "ign_f1": scores["ign_f1"], "precision": scores["precision"],
           "recall": scores["recall"], "detection_f1": scores["detection_f1"],
           "predicted": scores["predicted_facts"], "correct": round(scores["precision"] * scores["predicted_facts"] / 100)}
    row.update(extra or {})
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="<run folder>:<seed> whose scores are rescored")
    parser.add_argument("--members", nargs="*", default=[], help="<run folder>:<seed> of its members")
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--variants", nargs="+",
                        default=["scores", "scores+pair", "scores+pair+inverse", "scores+pair+inverse+types",
                                 "scores+pair+inverse+types+chain", "scores+pair+inverse+types+chain+doc",
                                 "scores+pair+inverse+types+chain+doc+corpus",
                                 "scores+pair+inverse+types+chain+doc+corpus+prior"])
    parser.add_argument("--out", default="results/rescore/report_dev.json")
    args = parser.parse_args()

    train, dev = load_split(args.data_dir, "train"), load_split(args.data_dir, "dev")
    stats = TrainStats(train)
    in_train = train_fact_names(train)
    gold = gold_facts(dev)
    scores = candidates(load_scores(args.run, "dev"), args.min_score)
    members = [load_scores(member, "dev") for member in args.members]
    print(json.dumps({"candidates": len(scores), "gold_covered": len(gold & set(scores)), "gold": len(gold),
                      "rules": sum(len(r) for r in stats.rules.values()),
                      "functional": sorted(r for r in stats.relation_count if stats.functional(r))}))

    rows = []
    threshold, _ = best_threshold(scores, gold)
    rows.append(report("base", dev, {f for f, s in scores.items() if s >= threshold}, in_train,
                       {"threshold": round(threshold, 3),
                        "f1_threshold_cross_fitted": threshold_cross_fitted(dev, scores, gold, args.folds)}))
    ruled = apply_rules(dev, scores, stats)
    threshold, _ = best_threshold(ruled, gold)
    rows.append(report("rules", dev, {f for f, s in ruled.items() if s >= threshold}, in_train,
                       {"threshold": round(threshold, 3),
                        "f1_threshold_cross_fitted": threshold_cross_fitted(dev, ruled, gold, args.folds)}))
    rows.append(report("buckets", dev, bucket_scores(dev, scores, gold, stats, args.folds), in_train))

    facts = sorted(scores)
    importance = {}
    for variant in args.variants:
        groups = variant.split("+")
        matrix, categorical, names = features(dev, [(f, scores[f]) for f in facts], members, stats, groups)
        rescored = cross_fitted(dev, facts, matrix, categorical, gold, args.folds)
        threshold, _ = best_threshold(rescored, gold)
        rows.append(report(f"rescore-{variant}", dev, {f for f, s in rescored.items() if s >= threshold},
                           in_train, {"threshold": round(threshold, 3),
                                      "f1_threshold_cross_fitted": threshold_cross_fitted(dev, rescored, gold, args.folds)}))
        importance[variant] = names
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"settings": vars(args), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
