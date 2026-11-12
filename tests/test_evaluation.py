import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from kg_extraction.evaluation.metrics import bcubed, clustering_scores, fact_scores  # noqa: E402


def test_perfect_clustering_ignores_cluster_names():
    s = clustering_scores(["a", "a", "b"], [7, 7, 3])
    assert s["b3_f1"] == pytest.approx(1) and s["v_f1"] == pytest.approx(1) and s["ari"] == pytest.approx(1)


def test_bcubed_single_cluster():
    # everything in one cluster: recall 1, precision = sum of squared class shares
    p, r, f1 = bcubed(["a", "a", "b", "b"], [0, 0, 0, 0])
    assert (p, r) == (pytest.approx(0.5), pytest.approx(1.0))


def test_fact_scores_micro_macro_and_no_relation():
    gold = {("d1", 0, "P17", 1), ("d1", 1, "P131", 2), ("d2", 0, "P17", 3)}
    pred = {("d1", 0, "P17", 1), ("d2", 0, "P131", 3), ("d2", 1, "no_relation", 2)}
    s = fact_scores(gold, pred)
    assert s["micro_p"] == pytest.approx(1 / 2) and s["micro_r"] == pytest.approx(1 / 3)
    assert s["per_relation"]["P17"]["f1"] == pytest.approx(2 / 3)
    assert s["per_relation"]["P131"]["f1"] == 0
    assert s["macro_f1"] == pytest.approx(1 / 3)


def _doc(i, rels):
    ents = [{"type": "PER", "mentions": [{"name": "Ann", "sent_id": 0, "pos": [0, 1]}]},
            {"type": "LOC", "mentions": [{"name": "Paris", "sent_id": 0, "pos": [3, 4]},
                                         {"name": "the city", "sent_id": 1, "pos": [0, 2]}]},
            {"type": "ORG", "mentions": [{"name": "Acme", "sent_id": 1, "pos": [3, 4]}]}]
    return {"id": i, "title": f"doc{i}", "sents": [["Ann", "lives", "in", "Paris", "."], ["the", "city", "hosts", "Acme"]],
            "entities": ents, "relations": rels}


@pytest.fixture
def tiny_redocred(tmp_path):
    raw = tmp_path / "data/raw/Linked-Re-DocRED"
    raw.mkdir(parents=True)
    rels = [{"r": "P551", "h": 0, "t": 1, "evidence": [0]}, {"r": "P159", "h": 2, "t": 1, "evidence": []}]
    for split, start in (("train", 0), ("dev", 100), ("test", 200)):
        (raw / f"{split}_revised.json").write_text(json.dumps([_doc(start + k, rels) for k in range(3)]))
    return tmp_path


def test_redocred_prepare_and_evaluate(tiny_redocred):
    from kg_extraction.data.redocred import prepare_redocred
    work = tiny_redocred
    cwd = os.getcwd()
    os.chdir(work)
    try:
        manifest = prepare_redocred()
    finally:
        os.chdir(cwd)
    assert manifest["dev"]["n_docs"] == 3 and manifest["dev"]["n_pairs"] == 18
    pairs = list(csv.DictReader(open(work / "data/processed/benchmark/redocred_dev_pairs.csv")))
    same = {(p["h"], p["t"]): p["same_sentence"] for p in pairs if p["doc_id"] == "100"}
    assert same[("0", "1")] == "1" and same[("0", "2")] == "0" and same[("2", "1")] == "1"
    docs = [json.loads(x) for x in open(work / "data/processed/benchmark/redocred_dev_docs.jsonl")]
    assert "relations" not in docs[0] and docs[0]["entities"][1]["name"] == "the city"

    run = work / "results/r"
    run.mkdir(parents=True)
    (run / "card.json").write_text(json.dumps({
        "run": "r", "author": "t", "method": "m", "supervision": "none",
        "training_data": "none", "models": "none", "description": "d"}))
    # doc 100: both facts right; doc 101: one right, one wrong; doc 102: nothing
    with open(run / "redocred_dev_facts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "h", "r", "t"])
        w.writerows([[100, 0, "P551", 1], [100, 2, "P159", 1], [101, 0, "P551", 1], [101, 0, "P159", 2]])
    out = subprocess.run([sys.executable, os.path.abspath(os.path.join(ROOT, "scripts/evaluate.py")), "results/r"],
                         cwd=work, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    s = json.load(open(run / "scores.json"))[0]
    assert s["micro_p"] == pytest.approx(3 / 4, abs=1e-4) and s["micro_r"] == pytest.approx(3 / 6, abs=1e-4)
    # evidence-only view: P159 facts (no evidence) are ignored on both sides
    assert s["evid_p"] == pytest.approx(2 / 3, abs=1e-4) and s["evid_r"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["detect_r"] == pytest.approx(3 / 6, abs=1e-4)
    # train has the same names, so the 3 hits are "seen in train": Ign precision 0 / (4 - 3)
    assert s["ign_p"] == 0 and s["ign_f1"] == 0
    # the 3 pairs related in gold and predicted related all get the right relation
    assert s["typing_f1"] == pytest.approx(1.0)
    assert s["f1_intra_sentence"] == pytest.approx(2 / 3, abs=1e-4)

    with open(run / "redocred_dev_facts.csv", "a", newline="") as f:
        csv.writer(f).writerow([999, 0, "P551", 1])
    bad = subprocess.run([sys.executable, os.path.abspath(os.path.join(ROOT, "scripts/evaluate.py")), "results/r"],
                         cwd=work, capture_output=True, text=True)
    assert bad.returncode != 0 and "not in subset" in bad.stderr


def test_redocred_seeds_graph_and_paired_bootstrap(tiny_redocred):
    from kg_extraction.data.redocred import prepare_redocred
    work = tiny_redocred
    cwd = os.getcwd()
    os.chdir(work)
    try:
        prepare_redocred()
    finally:
        os.chdir(cwd)
    script = os.path.abspath(os.path.join(ROOT, "scripts/evaluate.py"))
    for name, seeds in (("a", ([[100, 0, "P551", 1, 0.9], [101, 0, "P551", 1, 0.8]],
                               [[100, 0, "P551", 1, 0.9], [102, 0, "P551", 1, 0.7]])),
                        ("b", ([[100, 0, "P159", 1, 0.5]],))):
        run = work / f"results/{name}"
        run.mkdir(parents=True)
        (run / "card.json").write_text(json.dumps({
            "run": name, "author": "t", "method": "m", "supervision": "none",
            "training_data": "none", "models": "none", "description": "d"}))
        for seed, rows in enumerate(seeds):
            with open(run / f"redocred_dev_facts_seed{seed}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["doc_id", "h", "r", "t", "score"])
                w.writerows(rows)
        out = subprocess.run([sys.executable, script, f"results/{name}", "--leaderboard", "report/leaderboard.csv"],
                             cwd=work, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
    s = json.load(open(work / "results/a/scores.json"))[0]
    assert s["n_seeds"] == 2 and s["micro_f1_std"] == 0 and s["micro_f1_ci_low"] <= s["micro_f1"] <= s["micro_f1_ci_high"]
    # 2 right facts per seed, out of 3 gold P551 + 3 gold P159 facts, all doc-local nodes
    assert s["graph_p"] == 1 and s["graph_r"] == pytest.approx(2 / 6, abs=1e-4)
    assert (work / "report/leaderboard.md").exists()
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/compare_runs.py"), "results/a", "results/b"],
                         cwd=work, capture_output=True, text=True)
    assert out.returncode == 0 and "+50.00 points" in out.stdout, out.stdout + out.stderr


def test_eval_cell_matches_script(tiny_redocred):
    """notebooks/redocred_eval_cell.py (pasted into notebooks) must give the same scores as scripts/evaluate.py."""
    from kg_extraction.data.redocred import prepare_redocred
    work = tiny_redocred
    cwd = os.getcwd()
    os.chdir(work)
    try:
        prepare_redocred()
    finally:
        os.chdir(cwd)
    facts = [["100", 0, "P551", 1], ["100", 2, "P159", 1], ["101", 0, "P551", 1], ["101", 0, "P159", 2]]
    clusters = [["100", 0, 1, "a"], ["100", 2, 1, "b"], ["101", 0, 1, "a"], ["101", 2, 1, "a"]]
    run = work / "results/r"
    run.mkdir(parents=True)
    (run / "card.json").write_text(json.dumps({
        "run": "r", "author": "t", "method": "m", "supervision": "none",
        "training_data": "none", "models": "none", "description": "d"}))
    for name, header, rows in (("facts", ["doc_id", "h", "r", "t"], facts),
                               ("clusters", ["doc_id", "h", "t", "cluster"], clusters)):
        with open(run / f"redocred_dev_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    out = subprocess.run([sys.executable, os.path.abspath(os.path.join(ROOT, "scripts/evaluate.py")), "results/r"],
                         cwd=work, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    script = json.load(open(run / "scores.json"))[0]

    os.chdir(work)
    try:
        ns = {}
        exec(open(os.path.join(ROOT, "notebooks/redocred_eval_cell.py"), encoding="utf-8").read(), ns)
        cell = ns["evaluate_redocred"]([tuple(x) for x in facts], "dev", clusters=clusters, verbose=False)
    finally:
        os.chdir(cwd)
    shared = [k for k in script if k in cell and isinstance(cell[k], (int, float))]
    assert len(shared) >= 20
    for k in shared:
        assert cell[k] == pytest.approx(script[k], abs=1e-4), k
