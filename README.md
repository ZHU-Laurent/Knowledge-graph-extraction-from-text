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

See [docs/project_plan.md](docs/project_plan.md) for the plan, methods and evaluation, and [docs/personal_researches.md](docs/personal_researches.md) for the dataset selection. The first working version of the methods is on the branch `known-entities-v1`.

Every method is scored with the same benchmark and evaluator: see [docs/collaboration.md](docs/collaboration.md) for the rules and [docs/evaluation.md](docs/evaluation.md) for how to score a method (command-line script or a cell to paste into a notebook).

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
├── scripts/              Benchmark preparation, evaluation and baselines
├── src/kg_extraction/
│   ├── data/             Dataset loading and benchmark subsets
│   ├── pairs/            Candidate entity pairs
│   ├── relations/        Relation methods
│   └── evaluation/       Shared metrics
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

- Shared benchmark, evaluator and type-pair baseline: on `main`.
- PromptORE + clusters, PromptORE + classifier, ATLOP-style classifier and graph construction: on the branch `known-entities-v1`. Test F1 57.76 for the ATLOP-style classifier, 50.45 and 46.37 for the two PromptORE methods.
