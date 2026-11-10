# Knowledge graph extraction from text

Text mining project: find the relations between **known entities** of Linked-Re-DocRED documents, compare relations discovered without labels using [PromptORE](https://arxiv.org/abs/2304.01209) with supervised methods, and assemble the predicted facts into one knowledge graph.

## Pipeline

```text
Documents with known entities (Linked-Re-DocRED): mentions, types, coreference, Wikidata links
  → candidate ordered entity pairs
  → relation methods: type-pair baseline, PromptORE + clusters, PromptORE + classifier, ATLOP-style classifier
  → predicted facts with a score, or "no relation"
  → knowledge graph of the corpus: entities merged by Wikidata item, one edge per distinct fact
  → shared evaluation of the facts and of the graph
```

See [docs/project_plan.md](docs/project_plan.md) for the plan, methods and evaluation, [docs/known_entities_v1.md](docs/known_entities_v1.md) for the first version of the methods, and [docs/personal_researches.md](docs/personal_researches.md) for the dataset selection.

Every method is scored with the same benchmark and evaluator: see [docs/collaboration.md](docs/collaboration.md) for the rules and [docs/evaluation.md](docs/evaluation.md) for how to score a method (command-line script or a cell to paste into a notebook).

### Run

```bash
PYTHONPATH=src python scripts/train_relation_model.py
PYTHONPATH=src python scripts/build_graph.py --split dev
PYTHONPATH=src python scripts/build_graph.py --split test
PYTHONPATH=src python scripts/compare_methods.py
```

The first script encodes train and dev once, caches the features in `data/processed/features/`, trains the classifier and keeps the epoch with the best dev F1. The second writes the predictions, the graph (JSON and GraphML) and a report to `results/known_entities/`. The third scores the two PromptORE methods on the same splits.

## Project structure

```text
├── data/
│   ├── raw/              Downloaded datasets (not versioned)
│   └── processed/        Benchmark files, embeddings and models (not versioned)
├── docs/                 Subject, plan, dataset selection and evaluation rules
├── external/             Original PromptORE and Linked-DocRED code (not versioned)
├── notebooks/            Notebook version of the shared evaluator
├── report/               Leaderboard, final report and figures
├── results/              Predictions and scores of each run (not versioned)
├── scripts/              Benchmark preparation, evaluation, training and graph building
├── src/kg_extraction/
│   ├── data/             Dataset loading and benchmark subsets
│   ├── pairs/            Candidate entity pairs
│   ├── relations/        Relation methods: PromptORE and the ATLOP-style classifier
│   ├── graph/            Corpus graph assembly
│   └── evaluation/       Shared metrics, relation and graph metrics
└── tests/                Unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_benchmark.py
```

The original PromptORE code needs older libraries (Python 3.10, Transformers 4.26), so run it in a separate environment.

## Datasets

Data files are not stored in this repository. Download them into `data/raw/`:

| Role | Dataset |
|---|---|
| Main dataset | [Linked-Re-DocRED](https://github.com/alteca/Linked-DocRED/tree/main/Linked-Re-DocRED), into `data/raw/Linked-Re-DocRED/` |
| Reproduce the PromptORE paper | [FewRel](https://github.com/thunlp/FewRel), downloaded by `scripts/prepare_benchmark.py` |

## Status

- Shared benchmark, evaluator and type-pair baseline.
- PromptORE + clusters, PromptORE + classifier, ATLOP-style classifier and graph construction. Test F1 57.76 for the ATLOP-style classifier, 50.45 and 46.37 for the two PromptORE methods, measured with the metrics of `evaluation/relations.py`; they still have to be scored with the shared evaluator.
