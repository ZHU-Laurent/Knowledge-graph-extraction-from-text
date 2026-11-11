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


def test_cluster_typer_maps_clusters_to_relation_shares():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from kg_extraction.relations.clustering import ClusterTyper

    embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    labels = [["P19"], ["P19", "P17"], ["P17"], ["P17"]]
    typer = ClusterTyper(["P17", "P19"], n_clusters=2).fit(embeddings, labels)
    assert typer.scores(np.array([[0.05, 0.0]])).tolist() == [[0.5, 1.0]]
    assert typer.scores(np.array([[10.0, 10.1]])).tolist() == [[1.0, 0.0]]


def test_typing_scores_ignore_unrelated_pairs(docs):
    from kg_extraction.evaluation.relations import typing_scores

    predicted = {(0, 0, 1, "P19"), (0, 0, 2, "P17"), (0, 1, 2, "P17")}
    scores = typing_scores(docs, predicted)
    assert scores["typing_precision"] == 50.0
    assert scores["typing_recall"] == 50.0


def test_doc_model_scores_every_pair_with_each_option(docs):
    torch = pytest.importorskip("torch")
    from kg_extraction.relations.doc_model import (
        DocRelationModel, adaptive_focal_loss, predict_scores, prepare_document)

    torch.manual_seed(0)
    features = {"states": torch.randn(12, 16).half(), "attention": torch.rand(3, 12).half(),
                "positions": [[0], [5], [8]]}
    relations = ["P19", "P569"]
    item = prepare_document(docs[0], features, ["LOC", "PER", "TIME"], relations)
    assert item["labels"].shape == (6, 3)
    assert item["distances"].tolist() == [0] * 6
    plain = DocRelationModel(2, hidden=16, n_types=3, embedding=16, block=4)
    full = DocRelationModel(2, hidden=16, n_types=3, token_layers=1, pair_layers=1,
                            pair_features=True, embedding=16, block=4, pair_width=8)
    for model in (plain, full):
        logits = model(item)
        assert logits.shape == (6, 3)
        adaptive_focal_loss(logits, item["labels"].float()).backward()
    facts = predict_scores(plain, [item], docs[:1], relations, min_score=0.0)
    assert len(facts) == 12 and all(0 <= score <= 1 for score in facts.values())


def test_doc_model_trains_the_last_encoder_layers(docs):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from kg_extraction.relations.doc_model import DocRelationModel, prepare_document

    torch.manual_seed(0)
    config = transformers.BertConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                                     intermediate_size=32, attn_implementation="eager")
    layers = transformers.BertModel(config).encoder.layer
    # Two overlapping windows over 12 tokens, each with its [CLS] and [SEP] states.
    features = {"lower": [torch.randn(10, 16).half(), torch.randn(10, 16).half()],
                "windows": [(0, 8), (4, 12)], "length": 12, "positions": [[0], [5], [8]]}
    item = prepare_document(docs[0], features, ["LOC", "PER", "TIME"], ["P19", "P569"])
    model = DocRelationModel(2, hidden=16, n_types=3, embedding=16, block=4, top_layers=layers)
    states, attention = model.encoder(item)
    assert states.shape == (12, 16) and attention.shape == (3, 12)
    assert torch.allclose(attention.sum(1), torch.ones(3), atol=1e-4)
    model(item).sum().backward()
    assert layers[0].output.dense.weight.grad.abs().sum() > 0


def test_best_threshold_keeps_the_facts_that_maximize_f1():
    pytest.importorskip("torch")
    from kg_extraction.relations.doc_model import best_threshold

    facts = {(0, 0, 1, "P19"): 0.9, (0, 0, 2, "P569"): 0.6, (0, 1, 2, "P17"): 0.4,
             (0, 2, 1, "P17"): 0.1}
    gold = {(0, 0, 1, "P19"), (0, 0, 2, "P569")}
    assert best_threshold(facts, gold) == (0.6, 100.0)
