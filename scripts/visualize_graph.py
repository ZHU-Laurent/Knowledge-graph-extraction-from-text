"""Write an HTML viewer comparing the gold graph of a split with the graphs of some runs.

Each run is a folder written for the shared evaluator (redocred_<split>_facts_seed<N>.csv),
so any method works, ensembles included. Open the page in a browser: it needs no server
(vis-network is loaded from a CDN).

    python scripts/visualize_graph.py --split dev \
        --runs results/doc-tune-ens+old results/atlop-frozen results/promptore-clusters
"""
import argparse
from pathlib import Path

from kg_extraction.data.linked_docred import load_relation_names, load_split
from kg_extraction.evaluation.runs import read_run_facts
from kg_extraction.graph.viewer import render_viewer, viewer_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--runs", nargs="+", required=True, help="run folders to compare")
    parser.add_argument("--seed", type=int, default=0, help="seed of the run files")
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--output", help="default: results/graph_viewer_<split>.html")
    args = parser.parse_args()

    docs = load_split(args.data_dir, args.split)
    runs = {Path(run).name: read_run_facts(run, args.split, args.seed, docs) for run in args.runs}
    data = viewer_data(docs, load_relation_names(args.data_dir), runs, args.split)

    output = Path(args.output or f"results/graph_viewer_{args.split}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_viewer(data), encoding="utf-8")
    for run in data["runs"]:
        print(f"{run['name']}: fact F1 {run['facts']['f1']}, graph F1 {run['graph']['f1']}")
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
