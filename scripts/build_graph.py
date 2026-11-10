"""Predict the relations of a split with known entities and build its knowledge graph.

Writes the predictions in the Linked-DocRED format, the graph as JSON and GraphML,
and a report with the relation metrics and the graph metrics.
"""
import argparse
import copy
import json
from pathlib import Path

import networkx as nx
import torch

from kg_extraction.data.linked_docred import load_relation_names, load_split
from kg_extraction.evaluation.graph import graph_scores, graph_summary
from kg_extraction.evaluation.relations import relation_scores, train_fact_names
from kg_extraction.graph.build import build_graph, to_networkx
from kg_extraction.relations.doc_encoder import cached_features, default_device
from kg_extraction.relations.pair_classifier import PairClassifier, corpus_inputs, predict


def with_predictions(docs, facts):
    """Return a copy of the documents whose relations are the predicted facts, with scores."""
    predicted = copy.deepcopy(docs)
    for doc in predicted:
        doc["relations"] = []
    for (index, head, tail, relation), score in sorted(facts.items()):
        predicted[index]["relations"].append(
            {"h": head, "t": tail, "r": relation, "score": round(score, 6)})
    return predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--model", default="data/processed/models/relation_model.pt")
    parser.add_argument("--output-dir", default="results/known_entities")
    args = parser.parse_args()

    docs = load_split(args.data_dir, args.split)
    relation_names = load_relation_names(args.data_dir)
    features = cached_features(Path(args.features_dir) / f"{args.split}.pt", docs)
    checkpoint = torch.load(args.model, weights_only=False)
    relations = checkpoint["relations"]
    model = PairClassifier(len(relations)).to(default_device())
    model.load_state_dict(checkpoint["state"])
    pairs, inputs = corpus_inputs(features, docs)
    facts = predict(model, pairs, inputs, docs, relations)

    graph = build_graph(docs, facts, relation_names)
    report = {
        "split": args.split,
        "relations": relation_scores(docs, facts, train_fact_names(
            load_split(args.data_dir, "train"))),
        "graph": graph_scores(docs, graph),
        "summary": graph_summary(graph),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.split}_predictions.json").write_text(
        json.dumps(with_predictions(docs, facts)), encoding="utf-8")
    (output / f"{args.split}_graph.json").write_text(json.dumps(graph, indent=1),
                                                     encoding="utf-8")
    nx.write_graphml(to_networkx(graph), output / f"{args.split}_graph.graphml")
    (output / f"{args.split}_report.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
