"""Compare three relation methods with known entities, then the graphs they produce.

- atlop: the ATLOP-style classifier on frozen document features, trained by
  scripts/train_relation_model.py. Its threshold is learned, not set on dev.
- promptore_classifier: PromptORE [MASK] embeddings, a logistic detector and one
  logistic classifier per relation. Fact score: P(related) x P(relation | related).
- promptore_clusters: the same embeddings and detector, then k-means on related train
  pairs, each cluster mapped to the share of its train pairs that have each relation.
  Fact score: P(related) x share of the relation in the pair's cluster.

Both PromptORE methods take their fact threshold, and the clusters their k, from dev.
Test is scored once, with the settings chosen on dev.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from kg_extraction.data.linked_docred import load_relation_names, load_split
from kg_extraction.evaluation.graph import graph_scores, graph_summary
from kg_extraction.evaluation.relations import (
    gold_facts, relation_scores, train_fact_names, typing_scores)
from kg_extraction.graph.build import build_graph
from kg_extraction.pairs.candidates import allowed_type_pairs
from kg_extraction.relations.cache import cached_embeddings
from kg_extraction.relations.clustering import ClusterTyper
from kg_extraction.relations.doc_encoder import cached_features, default_device
from kg_extraction.relations.pair_classifier import PairClassifier, corpus_inputs, predict
from kg_extraction.relations.probe import (
    RelationProbe, best_threshold, facts_above, pair_instances, pair_list, sample_train_pairs)

SPLITS = ("dev", "test")


def atlop_facts(args, docs, relations):
    checkpoint = torch.load(args.atlop_model, weights_only=False)
    model = PairClassifier(len(checkpoint["relations"])).to(default_device())
    model.load_state_dict(checkpoint["state"])
    facts = {}
    for split in SPLITS:
        features = cached_features(Path(args.features_dir) / f"{split}.pt", docs[split])
        pairs, inputs = corpus_inputs(features, docs[split])
        facts[split] = predict(model, pairs, inputs, docs[split], checkpoint["relations"])
    return facts


def promptore_embeddings(args, docs, allowed):
    """Return the train sample and the dev and test pairs, with their PromptORE embeddings."""
    directory = Path(args.embeddings_dir)
    train_pairs = pair_list(docs["train"], allowed)
    kept, negative_weight = sample_train_pairs(train_pairs, args.negatives_per_positive,
                                               args.seed)
    train_pairs = [train_pairs[index] for index in kept]
    data = {"train": (train_pairs, cached_embeddings(
        directory / "train_sample.npy", pair_instances(docs["train"], train_pairs)))}
    for split in SPLITS:
        pairs = pair_list(docs[split], allowed)
        data[split] = (pairs, cached_embeddings(directory / f"{split}_isolated.npy",
                                                pair_instances(docs[split], pairs)))
    return data, negative_weight


def thresholded(data, docs, scores, relations):
    """Choose the fact threshold on dev, then return the facts of every split.

    scores holds the (pairs, relations) score matrix of each split.
    """
    threshold = best_threshold(data["dev"][0], scores["dev"], relations, gold_facts(docs["dev"]))
    facts = {split: facts_above(data[split][0], scores[split], relations, threshold)
             for split in SPLITS}
    return facts, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/linked-re-docred")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--embeddings-dir", default="data/processed/embeddings")
    parser.add_argument("--atlop-model", default="data/processed/models/relation_model.pt")
    parser.add_argument("--probe", default="data/processed/models/promptore_probe.joblib")
    parser.add_argument("--output-dir", default="results/comparison")
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--k-values", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    docs = {split: load_split(args.data_dir, split) for split in ("train", *SPLITS)}
    relation_names = load_relation_names(args.data_dir)
    relations = sorted(relation_names)
    train_names = train_fact_names(docs["train"])
    methods, settings = {}, {}

    methods["atlop"] = atlop_facts(args, docs, relations)
    print("atlop done", flush=True)

    data, negative_weight = promptore_embeddings(args, docs, allowed_type_pairs(docs["train"]))
    train_pairs, train_embeddings = data["train"]
    train_labels = [pair[3] for pair in train_pairs]
    if Path(args.probe).exists():
        probe = joblib.load(args.probe)
    else:
        probe = RelationProbe(relations).fit(train_embeddings, train_labels, negative_weight)
        Path(args.probe).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(probe, args.probe)
    methods["promptore_classifier"], threshold = thresholded(
        data, docs, {split: probe.scores(data[split][1]) for split in SPLITS}, relations)
    settings["promptore_classifier"] = {"threshold": threshold}
    print("promptore_classifier done", settings["promptore_classifier"], flush=True)

    related = np.array([bool(codes) for codes in train_labels])
    related_labels = [codes for codes in train_labels if codes]
    detection = {split: probe.detector.predict_proba(data[split][1])[:, 1] for split in SPLITS}
    best = None
    for k in args.k_values:
        typer = ClusterTyper(relations, k, args.seed).fit(train_embeddings[related],
                                                          related_labels)
        cluster_scores = {split: detection[split][:, None] * typer.scores(data[split][1])
                          for split in SPLITS}
        facts, threshold = thresholded(data, docs, cluster_scores, relations)
        dev_f1 = relation_scores(docs["dev"], facts["dev"], train_names)["f1"]
        print(f"clusters k={k}: dev F1 {dev_f1}, threshold {threshold:.2f}", flush=True)
        if best is None or dev_f1 > best[0]:
            best = (dev_f1, facts, {"k": k, "threshold": threshold})
    methods["promptore_clusters"], settings["promptore_clusters"] = best[1], best[2]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"settings": settings, "results": {}}
    for method, facts in methods.items():
        for split in SPLITS:
            graph = build_graph(docs[split], facts[split], relation_names)
            report["results"][f"{method}/{split}"] = {
                "relations": {**relation_scores(docs[split], facts[split], train_names),
                              **typing_scores(docs[split], facts[split])},
                "graph": graph_scores(docs[split], graph),
                "summary": {key: value for key, value in graph_summary(graph).items()
                            if key in ("nodes", "edges", "components")},
            }
            (output / f"{method}_{split}_graph.json").write_text(json.dumps(graph),
                                                                  encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    columns = ("precision", "recall", "f1", "ign_f1", "detection_f1", "typing_f1")
    print("| Method | Split | " + " | ".join(columns) + " | Graph F1 | P@1000 |")
    for name, result in report["results"].items():
        method, split = name.split("/")
        row = [result["relations"][column] for column in columns]
        row += [result["graph"]["f1"], result["graph"].get("precision_at_1000", "")]
        print(f"| {method} | {split} | " + " | ".join(map(str, row)) + " |")


if __name__ == "__main__":
    main()
