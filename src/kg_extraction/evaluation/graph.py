"""Score a predicted corpus graph against the gold graph built the same way."""
from collections import Counter

from kg_extraction.evaluation.relations import gold_facts
from kg_extraction.graph.build import build_graph


def triples(graph):
    return {(edge["head"], edge["relation"], edge["tail"]) for edge in graph["edges"]}


def graph_scores(docs, graph, ranks=(100, 500, 1000, 5000)):
    """Return graph fact precision, recall and F1, precision at N and precision by support."""
    gold = triples(build_graph(docs, dict.fromkeys(gold_facts(docs), 1.0)))
    predicted = triples(graph)
    correct = len(predicted & gold)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if correct else 0.0
    scores = {"precision": precision, "recall": recall, "f1": f1}

    ranked = [(edge["head"], edge["relation"], edge["tail"]) in gold for edge in graph["edges"]]
    for rank in ranks:
        if rank <= len(ranked):
            scores[f"precision_at_{rank}"] = sum(ranked[:rank]) / rank

    for name, keep in (("1", lambda n: n == 1), ("2", lambda n: n == 2), ("3+", lambda n: n >= 3)):
        hits = [is_gold for edge, is_gold in zip(graph["edges"], ranked) if keep(edge["support"])]
        if hits:
            scores[f"precision_support_{name}"] = sum(hits) / len(hits)
    scores = {name: round(100 * value, 2) for name, value in scores.items()}
    scores.update({"predicted_triples": len(predicted), "gold_triples": len(gold)})
    return scores


def graph_summary(graph, top=10):
    """Return the size of the graph, its most connected nodes and its most frequent relations."""
    import networkx as nx

    from kg_extraction.graph.build import to_networkx

    network = to_networkx(graph)
    components = sorted((len(part) for part in nx.weakly_connected_components(network)),
                        reverse=True)
    labels = {node["id"]: node["label"] for node in graph["nodes"]}
    degrees = sorted(network.degree(), key=lambda item: -item[1])[:top]
    relations = Counter(edge["relation_name"] for edge in graph["edges"])
    return {"nodes": network.number_of_nodes(), "edges": network.number_of_edges(),
            "linked_nodes": sum(bool(node["wikidata"]) for node in graph["nodes"]),
            "components": len(components), "largest_component": components[0] if components else 0,
            "top_nodes": [{"id": key, "label": labels[key], "degree": degree}
                          for key, degree in degrees],
            "top_relations": relations.most_common(top)}
