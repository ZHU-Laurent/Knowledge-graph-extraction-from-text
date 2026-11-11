"""Assemble document-level facts into one knowledge graph for the corpus.

Nodes merge across documents: entities with a Wikidata item merge on that item,
dates and numbers merge on their normalized text, and the other entities stay
local to their document. A fact stated in several documents becomes one edge,
whose confidence is the noisy-or of its scores in each document.
"""
import re
from collections import Counter, defaultdict

LITERAL_TYPES = ("TIME", "NUM")
WIKIDATA_ITEM = re.compile(r"Q\d+")


def normalize_literal(text):
    """Return a merge key for a date or a number: lower case, punctuation and spaces removed."""
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def node_key(doc, entity):
    """Return the identifier of the graph node an entity belongs to."""
    # Dates and numbers carry "#ignored#", and entities without an item "#DocRED-<n>#".
    link = (entity.get("entity_linking") or {}).get("wikidata_resource") or ""
    if WIKIDATA_ITEM.fullmatch(link):
        return f"wd:{link}"
    if entity["type"] in LITERAL_TYPES:
        names = [normalize_literal(mention["name"]) for mention in entity["mentions"]]
        return f"{entity['type'].lower()}:{max(names, key=len)}"
    return f"doc:{doc['title']}#{entity.get('id')}"


def build_graph(docs, facts, relation_names=None):
    """Return the corpus graph of facts, as {"nodes": [...], "edges": [...]}.

    facts maps (document index, head id, tail id, relation code) to a score in [0, 1].
    """
    relation_names = relation_names or {}
    # Entity ids are their positions in the document (Linked-DocRED also stores them as "id").
    keys = [{position: node_key(doc, {"id": position, **entity})
             for position, entity in enumerate(doc["entities"])} for doc in docs]

    # Within a document, two entities can share a node: keep the best score of the fact.
    support = defaultdict(dict)
    for (index, head, tail, relation), score in facts.items():
        triple = (keys[index][head], relation, keys[index][tail])
        if triple[0] == triple[2]:
            continue
        support[triple][index] = max(score, support[triple].get(index, 0.0))

    edges = []
    for (head, relation, tail), scores in support.items():
        miss = 1.0
        for score in scores.values():
            miss *= 1.0 - score
        edges.append({"head": head, "relation": relation,
                      "relation_name": relation_names.get(relation, relation), "tail": tail,
                      "score": round(1.0 - miss, 6), "max_score": round(max(scores.values()), 6),
                      "support": len(scores),
                      "documents": [{"title": docs[index]["title"], "score": round(score, 6)}
                                    for index, score in sorted(scores.items())]})
    edges.sort(key=lambda edge: (-edge["score"], edge["head"], edge["relation"], edge["tail"]))

    used = {edge["head"] for edge in edges} | {edge["tail"] for edge in edges}
    names, types, titles = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    for index, doc in enumerate(docs):
        for position, entity in enumerate(doc["entities"]):
            key = keys[index][position]
            if key not in used:
                continue
            names[key].update(mention["name"] for mention in entity["mentions"])
            types[key][entity["type"]] += 1
            titles[key].add(doc["title"])
    nodes = [{"id": key, "label": names[key].most_common(1)[0][0],
              "type": types[key].most_common(1)[0][0],
              "wikidata": key[3:] if key.startswith("wd:") else None,
              "documents": sorted(titles[key])} for key in sorted(used)]
    return {"nodes": nodes, "edges": edges}


def to_networkx(graph):
    """Return the graph as a networkx MultiDiGraph, one edge per fact."""
    import networkx as nx

    result = nx.MultiDiGraph()
    for node in graph["nodes"]:
        result.add_node(node["id"], label=node["label"], type=node["type"],
                        wikidata=node["wikidata"] or "", n_documents=len(node["documents"]))
    for edge in graph["edges"]:
        result.add_edge(edge["head"], edge["tail"], key=edge["relation"],
                        relation=edge["relation"], relation_name=edge["relation_name"],
                        score=edge["score"], support=edge["support"])
    return result
