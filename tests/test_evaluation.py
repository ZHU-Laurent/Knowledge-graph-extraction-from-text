import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from kg_extraction.data.benchmark import fewrel_instances, prepare_fewrel  # noqa: E402
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


@pytest.fixture
def tiny_fewrel(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    data = {"P1": [{"tokens": "A met B".split(), "h": ["A", "Q1", [[0]]], "t": ["B", "Q2", [[2]]]}] * 2,
            "P2": [{"tokens": "C in D E".split(), "h": ["C", "Q3", [[0]]], "t": ["D E", "Q4", [[2, 3]]]}] * 3}
    for f in ("val_nyt.json", "val_pubmed.json"):
        (raw / f).write_text(json.dumps(data))
    return tmp_path


def test_prepare_fewrel_hides_labels(tiny_fewrel):
    out = tiny_fewrel / "bench"
    prepare_fewrel(str(tiny_fewrel / "raw"), str(out), names=("nyt",))
    lines = [json.loads(x) for x in (out / "nyt.jsonl").read_text().splitlines()]
    assert [x["id"] for x in lines] == [f"nyt-{i:05d}" for i in range(5)]
    assert "relation" not in lines[0] and lines[2]["t"] == {"name": "D E", "start": 2, "end": 4}
    gold = list(csv.DictReader(open(out / "nyt_gold.csv")))
    assert [g["relation"] for g in gold] == ["P1", "P1", "P2", "P2", "P2"]


def test_evaluate_script_end_to_end(tiny_fewrel, tmp_path):
    work = tmp_path / "work"
    bench = work / "data/processed/benchmark"
    prepare_fewrel(str(tiny_fewrel / "raw"), str(bench), names=("nyt",))
    run = work / "results/demo"
    run.mkdir(parents=True)
    (run / "card.json").write_text(json.dumps({
        "run": "demo", "author": "test", "method": "toy", "supervision": "none",
        "training_data": "none", "models": "none", "description": "toy run"}))
    for seed, clusters in ((0, "xxyyy"), (1, "xyyyy")):
        with open(run / f"fewrel_nyt_k_true_seed{seed}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "cluster"])
            w.writerows([f"nyt-{i:05d}", c] for i, c in enumerate(clusters))
    out = subprocess.run([sys.executable, os.path.abspath(os.path.join(ROOT, "scripts/evaluate.py")),
                          "results/demo", "--leaderboard", "report/leaderboard.csv"],
                         cwd=work, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    rows = list(csv.DictReader(open(work / "report/leaderboard.csv")))
    assert len(rows) == 1 and rows[0]["n_seeds"] == "2" and float(rows[0]["b3_f1"]) < 1
