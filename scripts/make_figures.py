"""Figures of the report and the slides, written to report/figures/ (PNG, 200 dpi).

Scores come from report/leaderboard.csv, results/calibration/calibration.json
(scripts/calibration.py) and results/label_efficiency/ (scripts/label_efficiency.py and
train_doc_model.py --train-fraction). The gain of each change is copied from
docs/analysis_v2.md, where each value comes from one run. A figure whose inputs are
missing is skipped.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kg_extraction.data.linked_docred import load_split  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
METHODS = [  # run, label, family color
    ("baseline-typepair", "M0 type-pair baseline", GRAY),
    ("m0-structural-v2", "M0' structural baseline", GRAY),
    ("promptore-clusters", "M1 PromptORE + clusters", ORANGE),
    ("promptore-classifier", "M2 PromptORE + classifier", ORANGE),
    ("atlop-frozen", "M3 ATLOP-style, frozen", BLUE),
    ("doc-full", "Document model, frozen", BLUE),
    ("doc-tune4", "Document model, 4 layers tuned", BLUE),
    ("doc-tune-ens+old", "Final ensemble", BLUE),
]
# Dev F1, seed 0 (docs/analysis_v2.md, sections 2, 5 and 6).
PROGRESSION = [
    ("ATLOP-style head,\nfrozen BERT (v1)", 58.37),
    ("threshold\nchosen on dev", 63.15),
    ("+ pair features,\ntoken and pair layers", 65.99),
    ("ensemble of\n3 seeds", 67.68),
    ("+ DeBERTa-v3-large\nmodel", 69.94),
    ("+ PromptORE\nclassifier (M2)", 70.48),
    ("last 4 encoder\nlayers trained", 73.60),
    ("final ensemble", 75.05),
]
# Changes that did not help, gain in dev F1 (analysis_v1.md section 2, analysis_v2.md section 2).
DROPPED = [
    ("Inverse and transitivity rules", -3.77),
    ("Adaptive focal loss", -0.43),
    ("One threshold per relation\n(cross-fitted)", -0.11),
    ("One threshold per frequency\nbucket (cross-fitted)", -0.32),
    ("Larger frozen encoder\n(roberta-large, plain head)", -2.11),
]


def style(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis=grid, color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def leaderboard(path, dataset):
    return {row["run"]: row for row in csv.DictReader(open(path, encoding="utf-8"))
            if row["dataset"] == dataset}


def dataset_figure(docs, out):
    counts = Counter(fact["r"] for doc in docs for fact in doc["relations"])
    n_pairs = sum(len(doc["entities"]) * (len(doc["entities"]) - 1) for doc in docs)
    related = len({(index, fact["h"], fact["t"])
                   for index, doc in enumerate(docs) for fact in doc["relations"]})
    values = sorted(counts.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(values)), values, color=BLUE, width=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Relations, most frequent first", color=MUTED)
    ax.set_ylabel("Train facts (log scale)", color=MUTED)
    ax.set_title(f"{len(values)} relations, {sum(values):,} train facts: a long tail",
                 loc="left", color=INK, fontsize=11)
    ax.text(0.98, 0.92, f"{len(docs):,} train documents\n{n_pairs:,} ordered entity pairs\n"
            f"{100 * (1 - related / n_pairs):.1f}% of pairs have no relation",
            transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=9)
    style(ax, "y")
    save(fig, out / "dataset_long_tail.png")


def methods_figure(board, out):
    rows = [(label, float(board[run]["micro_f1"]) * 100, float(board[run]["graph_f1"]) * 100, color)
            for run, label, color in METHODS if run in board]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    positions = range(len(rows))
    ax.barh(positions, [row[1] for row in rows], color=[row[3] for row in rows], height=0.6)
    for position, (label, f1, graph_f1, _) in zip(positions, rows):
        ax.text(f1 + 0.8, position, f"{f1:.1f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(list(positions), [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 85)
    ax.set_xlabel("Fact F1 on test (provisional)", color=MUTED)
    ax.set_title("Supervised document models beat PromptORE by 25 points",
                 loc="left", color=INK, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in (GRAY, ORANGE, BLUE)]
    ax.legend(handles, ["baseline", "PromptORE", "supervised"], frameon=False, fontsize=8,
              loc="upper right")
    style(ax)
    save(fig, out / "methods_test_f1.png")


def progression_figure(out):
    fig, (ax, side) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    labels, values = zip(*PROGRESSION)
    ax.plot(range(len(values)), values, color=BLUE, linewidth=2, marker="o", markersize=7)
    for index, value in enumerate(values):
        gain = f"\n+{value - values[index - 1]:.1f}" if index else ""
        ax.annotate(f"{value:.1f}{gain}", (index, value), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(labels)), [label.replace("\n", " ") for label in labels], fontsize=8,
                  rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(55, 79)
    ax.set_ylabel("Dev F1", color=MUTED)
    ax.set_title("What worked: dev F1 after each change", loc="left", color=INK, fontsize=11)
    style(ax, "y")

    names, gains = zip(*DROPPED)
    side.barh(range(len(gains)), gains, color=ORANGE, height=0.55)
    for index, gain in enumerate(gains):
        side.text(gain - 0.1, index, f"{gain:+.1f}", va="center", ha="right", fontsize=8,
                  color=INK)
    side.set_yticks(range(len(names)), names, fontsize=8)
    side.invert_yaxis()
    side.set_xlim(-5, 0.5)
    side.axvline(0, color=MUTED, linewidth=0.8)
    side.set_xlabel("Change in dev F1", color=MUTED)
    side.set_title("What did not (dropped)", loc="left", color=INK, fontsize=11)
    style(side)
    save(fig, out / "progression_dev_f1.png")


def detection_typing_figure(board, out):
    rows = [(label, float(board[run]["detect_f1"]) * 100, float(board[run]["typing_f1"]) * 100)
            for run, label, _ in METHODS if run in board]
    fig, ax = plt.subplots(figsize=(7, 4))
    for index, (label, detection, typing) in enumerate(rows):
        ax.plot([detection, typing], [index, index], color="#c9c8c3", linewidth=2, zorder=1)
    ax.scatter([row[1] for row in rows], range(len(rows)), color=ORANGE, s=50, zorder=2,
               label="detection F1: is the pair related?")
    ax.scatter([row[2] for row in rows], range(len(rows)), color=BLUE, s=50, zorder=2,
               label="typing F1: which relation, given a related pair?")
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(20, 100)
    ax.set_xlabel("F1 on test", color=MUTED)
    ax.set_title("Detection is the hard part, not typing", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)
    style(ax)
    save(fig, out / "detection_vs_typing.png")


def frequency_figure(board, out):
    runs = [method for method in METHODS[1:] if method[0] in ("promptore-clusters",
            "promptore-classifier", "atlop-frozen", "doc-tune-ens+old")]
    buckets = [("macro_f1_frequent", "10 frequent"), ("macro_f1_medium", "medium"),
               ("macro_f1_rare", "rare (< 200 train facts)")]
    colors = [BLUE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    width = 0.26
    for offset, ((column, name), color) in enumerate(zip(buckets, colors)):
        values = [float(board[run][column]) * 100 for run, _, _ in runs]
        positions = [index + (offset - 1) * width for index in range(len(runs))]
        ax.bar(positions, values, width=width - 0.03, color=color, label=name)
        for position, value in zip(positions, values):
            ax.text(position, value + 1, f"{value:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(runs)), [label.replace(" + ", "\n+ ").replace(", ", "\n") for _, label, _ in runs],
                  fontsize=8)
    ax.set_ylabel("Macro F1 on test", color=MUTED)
    ax.set_title("Rare relations stay the weak spot", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, title="relations", title_fontsize=8)
    style(ax, "y")
    save(fig, out / "relation_frequency.png")


def graph_support_figure(board, out):
    rows = [(label, [float(board[run][f"graph_precision_support_{n}"] or "nan") * 100
                     for n in ("1", "2", "3+")]) for run, label, _ in METHODS[1:]
            if run in ("promptore-classifier", "atlop-frozen", "doc-tune-ens+old")]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for (label, values), color in zip(rows, (ORANGE, GRAY, BLUE)):
        ax.plot(["1 document", "2 documents", "3 or more"], values, marker="o", markersize=7,
                linewidth=2, color=color, label=label)
        ax.annotate(f"{values[0]:.0f}", (0, values[0]), textcoords="offset points",
                    xytext=(-14, -3), fontsize=8, color=INK)
    ax.set_ylim(40, 102)
    ax.set_ylabel("Edge precision on test", color=MUTED)
    ax.set_xlabel("Documents stating the edge", color=MUTED)
    ax.set_title("An edge seen in several documents is almost always right",
                 loc="left", color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "y")
    save(fig, out / "graph_precision_by_support.png")


def calibration_figure(path, out):
    report = json.loads(path.read_text(encoding="utf-8"))
    shown = [("promptore-classifier", "M2 PromptORE + classifier", ORANGE),
             ("doc-tune4", "Document model, 4 layers tuned", AQUA),
             ("doc-tune-ens+old", "Final ensemble", BLUE)]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot([0, 1], [0, 1], color="#c9c8c3", linestyle="--", linewidth=1, label="perfectly calibrated")
    for run, label, color in shown:
        if run in report:
            bins = report[run]["facts"]["bins"]
            ax.plot([b["mean_score"] for b in bins], [b["precision"] for b in bins], marker="o",
                    markersize=6, linewidth=2, color=color,
                    label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean fact score in the bin", color=MUTED)
    ax.set_ylabel("Share of correct facts (dev)", color=MUTED)
    ax.set_title("Precision rises steadily with the score", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(ax, "both")
    save(fig, out / "calibration_dev.png")


def label_efficiency_figure(directory, out):
    series = {}
    promptore = directory / "promptore.json"
    if promptore.exists():
        for row in json.loads(promptore.read_text(encoding="utf-8")):
            series.setdefault(row["method"], {}).setdefault(row["fraction"], []).append(row["f1"])
    for report in directory.glob("le-doc-full-f*/train_report_seed*.json"):
        data = json.loads(report.read_text(encoding="utf-8"))
        series.setdefault("doc", {}).setdefault(data["settings"]["train_fraction"], []).append(
            data["best"]["dev_f1"])
    if "doc" in series:
        series["doc"][1.0] = [66.18]  # doc-full, mean of 3 seeds (analysis_v2.md section 5)
    if not series:
        return
    shown = [("M1", "M1 PromptORE + clusters", ORANGE), ("M2", "M2 PromptORE + classifier", YELLOW),
             ("doc", "Document model, frozen", BLUE)]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for key, label, color in shown:
        if key not in series:
            continue
        fractions = sorted(series[key])
        means = [sum(series[key][f]) / len(series[key][f]) for f in fractions]
        ax.plot([100 * f for f in fractions], means, marker="o", markersize=7, linewidth=2,
                color=color, label=label)
        for fraction, mean in zip(fractions, means):
            ax.annotate(f"{mean:.1f}", (100 * fraction, mean), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 100], ["5%", "10%", "25%", "50%", "100%"])
    ax.set_xlabel("Share of the 3,053 train documents with labels", color=MUTED)
    ax.set_ylabel("Dev F1", color=MUTED)
    ax.set_title("How many labels each method needs", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "y")
    save(fig, out / "label_efficiency_dev.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--leaderboard", default="report/leaderboard.csv")
    parser.add_argument("--calibration", default="results/calibration/calibration.json")
    parser.add_argument("--label-efficiency", default="results/label_efficiency")
    parser.add_argument("--output-dir", default="report/figures")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test = leaderboard(args.leaderboard, "redocred_test")
    dataset_figure(load_split(args.data_dir, "train"), out)
    methods_figure(test, out)
    progression_figure(out)
    detection_typing_figure(test, out)
    frequency_figure(test, out)
    graph_support_figure(test, out)
    if Path(args.calibration).exists():
        calibration_figure(Path(args.calibration), out)
    label_efficiency_figure(Path(args.label_efficiency), out)


if __name__ == "__main__":
    main()
