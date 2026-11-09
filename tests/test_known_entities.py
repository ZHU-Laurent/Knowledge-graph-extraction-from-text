import pytest

from kg_extraction.evaluation.graph import graph_scores
from kg_extraction.graph.build import build_graph, node_key


def entity(entity_id, entity_type, link, *mentions):
    return {"id": entity_id, "type": entity_type,
            "entity_linking": {"wikidata_resource": link},
            "mentions": [{"name": name, "sent_id": sent, "pos": [start, end]}
                         for name, sent, start, end in mentions]}


def document(title, entities, relations=()):
    return {"title": title, "sents": [["Ada", "Lovelace", "was", "born", "in", "London", "in",
                                       "1815", "."]],
            "entities": entities, "relations": [{"h": h, "t": t, "r": r} for h, t, r in relations]}


@pytest.fixture
def docs():
    first = document("A", [entity(0, "PER", "Q7259", ("Ada Lovelace", 0, 0, 2)),
                           entity(1, "LOC", "Q84", ("London", 0, 5, 6)),
                           entity(2, "TIME", "#ignored#", ("1815", 0, 7, 8))],
                     [(0, 1, "P19"), (0, 2, "P569")])
    second = document("B", [entity(0, "PER", "Q7259", ("Lovelace", 0, 1, 2)),
                            entity(1, "LOC", "Q84", ("London", 0, 5, 6)),
                            entity(2, "MISC", "#DocRED-12#", ("Analytical Engine", 0, 0, 1))],
                      [(0, 1, "P19")])
    return [first, second]


def test_node_keys_merge_links_and_literals(docs):
    assert node_key(docs[0], docs[0]["entities"][0]) == "wd:Q7259"
    assert node_key(docs[0], docs[0]["entities"][2]) == "time:1815"
    assert node_key(docs[1], docs[1]["entities"][2]) == "doc:B#2"


def test_facts_from_several_documents_merge_with_noisy_or(docs):
    graph = build_graph(docs, {(0, 0, 1, "P19"): 0.5, (1, 0, 1, "P19"): 0.5})
    assert len(graph["edges"]) == 1
    edge = graph["edges"][0]
    assert edge["score"] == pytest.approx(0.75)
    assert edge["support"] == 2
    assert {node["id"] for node in graph["nodes"]} == {"wd:Q7259", "wd:Q84"}


def test_self_loops_are_dropped(docs):
    docs[0]["entities"][1]["entity_linking"]["wikidata_resource"] = "Q7259"
    assert build_graph(docs, {(0, 0, 1, "P19"): 0.9})["edges"] == []


def test_graph_scores_against_gold(docs):
    graph = build_graph(docs, {(0, 0, 1, "P19"): 0.9, (0, 1, 0, "P17"): 0.6})
    scores = graph_scores(docs, graph, ranks=(1,))
    assert scores["gold_triples"] == 2
    assert scores["precision"] == 50.0
    assert scores["recall"] == 50.0
    assert scores["precision_at_1"] == 100.0


def test_adaptive_threshold_loss_prefers_gold_above_threshold():
    torch = pytest.importorskip("torch")
    from kg_extraction.relations.pair_classifier import adaptive_threshold_loss, fact_scores

    labels = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    good = torch.tensor([[0.0, 2.0, -2.0], [0.0, -2.0, -2.0]])
    bad = torch.tensor([[0.0, -2.0, 2.0], [0.0, 2.0, 2.0]])
    assert adaptive_threshold_loss(good, labels) < adaptive_threshold_loss(bad, labels)
    assert (fact_scores(good) > 0.5).tolist() == [[True, False], [False, False]]


def test_pair_labels_mark_unrelated_pairs(docs):
    pytest.importorskip("torch")
    from kg_extraction.relations.pair_classifier import document_pairs, pair_labels

    pairs = document_pairs(docs[0])
    labels = pair_labels(docs[0], pairs, ["P19", "P569"])
    assert len(pairs) == 6
    assert labels[:, 0].sum() == 4
    assert labels[0].tolist() == [0, 1, 0]


def test_markers_wrap_every_mention(docs):
    transformers = pytest.importorskip("transformers")
    from kg_extraction.relations.doc_encoder import tokenize_document, windows

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased",
                                                               local_files_only=True)
    except OSError:
        pytest.skip("bert-base-cased is not cached")
    ids, positions = tokenize_document(tokenizer, docs[0])
    tokens = tokenizer.convert_ids_to_tokens(ids)
    assert all(tokens[position] == "*" for entity in positions for position in entity)
    assert tokens.count("*") == 6
    assert windows(600, window=510, stride=256) == [(0, 510), (90, 600)]
