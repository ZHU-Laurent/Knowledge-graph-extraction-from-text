# Knowledge graph extraction from text

Text mining project: recover labeled relations between known entities by clustering [PromptORE](https://arxiv.org/abs/2304.01209) embeddings, then assigning each cluster to a fixed list of relations that includes "no relation".

## Pipeline

```text
Documents with gold entities (Linked-Re-DocRED)
  → candidate ordered entity pairs
  → relation embeddings (PromptORE prompt, untrained BERT)
  → clusters (k-means, no labels)
  → cluster-to-relation mapping (train labels, dev threshold)
  → predicted relations or "no relation"
  → evaluation and comparison with a linear classifier
```

See [docs/project_plan.md](docs/project_plan.md) for the design decisions, evaluation and work plan, and [docs/personal_researches.md](docs/personal_researches.md) for the dataset selection.

## Project structure

```text
├── configs/              Experiment settings
├── data/
│   ├── raw/              Downloaded datasets (not versioned)
│   └── processed/        Converted data and embeddings (not versioned)
├── docs/                 Subject, plan and dataset selection
├── external/             Original PromptORE and Linked-DocRED code (not versioned)
├── notebooks/            Exploration and error analysis
├── report/               Final report and figures
├── results/              Metrics and predictions (not versioned)
├── scripts/              Command-line entry points
├── src/kg_extraction/
│   ├── data/             Dataset loading
│   ├── pairs/            Candidate entity pairs
│   ├── relations/        Relation embeddings and clustering
│   ├── mapping/          Cluster-to-relation assignment
│   ├── baselines/        Comparison methods
│   └── evaluation/       Metrics and error analysis
└── tests/                Unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

These dependencies have not been installed or tested yet. The original PromptORE code was tested on Python 3.8, so run it in a separate environment.

## Datasets

Data files are not stored in this repository. Download them into `data/raw/`:

| Role | Dataset |
|---|---|
| Main dataset | [Linked-Re-DocRED](https://github.com/alteca/Linked-DocRED/tree/main/Linked-Re-DocRED) |
| Reproduce the PromptORE paper | [FewRel](https://github.com/thunlp/FewRel) |

## Status

Project structure and plan only: nothing is implemented yet.
