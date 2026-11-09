# Knowledge graph extraction from text

Text mining project: build a knowledge graph from documents whose entities are known.

**Branch `known-entities-v1`.** This branch does not follow the plan of `main`. It assumes the entities are given, with their mentions, types and Wikidata links, and only covers relation detection and graph construction. Its main method is an ATLOP-style classifier, compared with two methods built on PromptORE (classifier, and detector then clusters). See [docs/known_entities_v1.md](docs/known_entities_v1.md).

## Pipeline

```text
Documents with known entities (Linked-Re-DocRED)
  → document encoding, once per document (frozen BERT, mentions wrapped in markers)
  → relation classifier over every ordered entity pair (ATLOP-style head, trained on train)
  → document facts (head, relation, tail) with a score
  → corpus graph: nodes merged on Wikidata items, edges merged across documents
  → relation metrics and graph metrics
```

### Run

```bash
PYTHONPATH=src python scripts/train_relation_model.py
PYTHONPATH=src python scripts/build_graph.py --split dev
PYTHONPATH=src python scripts/build_graph.py --split test
PYTHONPATH=src python scripts/compare_methods.py
```

The first script encodes train and dev once, caches the features in `data/processed/features/`, trains the classifier and keeps the epoch with the best dev F1. The second writes the predictions, the graph (JSON and GraphML) and a report to `results/known_entities/`.

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
│   ├── relations/        Document encoder and relation classifier
│   ├── graph/            Corpus graph assembly
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
