# Knowledge graph extraction from text

Text mining project: find the relations between **known entities** of Linked-Re-DocRED documents, compare relations discovered without labels using [PromptORE](https://arxiv.org/abs/2304.01209) with supervised methods, and assemble the predicted facts into one knowledge graph.

## Pipeline

```text
Documents with known entities (Linked-Re-DocRED): mentions, types, coreference, Wikidata links
  → candidate ordered entity pairs
  → relation methods: type-pair baseline, PromptORE + clusters, PromptORE + classifier,
    ATLOP-style classifier, document model with a partly fine-tuned encoder, ensembles
  → predicted facts with a score, or "no relation"
  → knowledge graph of the corpus: entities merged by Wikidata item, one edge per distinct fact
  → shared evaluation of the facts and of the graph
```

See [docs/project_plan.md](docs/project_plan.md) for the plan, methods and evaluation, [docs/known_entities_v1.md](docs/known_entities_v1.md) for the first version of the methods, [docs/analysis_v1.md](docs/analysis_v1.md) and [docs/analysis_v2.md](docs/analysis_v2.md) for the results and what each change brings, and [docs/personal_researches.md](docs/personal_researches.md) for the dataset selection.

Every method is scored with the same benchmark and evaluator: see [docs/collaboration.md](docs/collaboration.md) for the rules and [docs/evaluation.md](docs/evaluation.md) for how to score a method (command-line script or a cell to paste into a notebook).

### Run

```bash
export PYTHONPATH=src
# first version of the methods: ATLOP-style classifier on a frozen encoder, PromptORE methods
python scripts/train_relation_model.py
python scripts/compare_methods.py
# best single model: document model with the last 4 encoder layers trained
python scripts/train_doc_model.py --run doc-tune4 --tune-layers 4 --pair-features --pair-layers 2 \
    --learning-rate 1e-4 --epochs 15 --seed 0
# score a run and add it to the leaderboard
python scripts/evaluate.py results/doc-tune4 --leaderboard report/leaderboard.csv
# knowledge graph of the facts kept by a run
python scripts/build_graph.py --run results/doc-tune4 --split dev
```

Encoder states are cached once in `data/processed/features/`; every script keeps the epoch and the fact threshold with the best dev F1. `build_graph.py` writes the predictions, the graph (JSON and GraphML) and a report to `<run>/graph/`. Ensembles are built with `scripts/combine_runs.py`; [docs/analysis_v2.md](docs/analysis_v2.md) lists the commands of every run.

## Project structure

```text
├── data/
│   ├── raw/              Downloaded datasets (not versioned)
│   └── processed/        Benchmark files, embeddings and models (not versioned)
├── docs/                 Plan, methods, dataset selection and evaluation rules
├── notebooks/            Notebook version of the shared evaluator
├── report/               Leaderboard, final report and figures
├── results/              Predictions and scores of each run (not versioned)
├── scripts/              Benchmark preparation, evaluation, training and graph building
├── src/kg_extraction/
│   ├── data/             Dataset loading and benchmark subsets
│   ├── pairs/            Candidate entity pairs
│   ├── relations/        Relation methods: PromptORE, the ATLOP-style classifier and the document model
│   ├── graph/            Corpus graph assembly
│   └── evaluation/       Shared metrics, relation and graph metrics
└── tests/                Unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_benchmark.py   # after downloading the dataset (below)
```

## Datasets

The project uses one dataset, so that every method is scored on the same documents: [Linked-Re-DocRED](https://github.com/alteca/Linked-DocRED/tree/main/Linked-Re-DocRED). Data files are not stored in this repository; download it into `data/raw/Linked-Re-DocRED/`.

## Status

Test scores of Linked-Re-DocRED from the shared evaluator ([report/leaderboard.md](report/leaderboard.md)). They are provisional: every choice was made on dev, and the final test run comes last.

| Method | Run | F1 | Ign F1 | Graph F1 |
|---|---|---|---|---|
| Type-pair baseline | `baseline-typepair` | 18.86 | 17.80 | 19.42 |
| PromptORE + clusters | `promptore-clusters` | 45.93 | 43.56 | 46.01 |
| PromptORE + classifier | `promptore-classifier` | 50.41 | 48.07 | 50.70 |
| ATLOP-style classifier, frozen encoder | `atlop-frozen` | 57.70 | 56.76 | 59.20 |
| Document model, frozen encoder | `doc-full` | 66.04 | 64.53 | 66.88 |
| Document model, last 4 encoder layers trained | `doc-tune4` | 73.55 | 72.27 | 74.34 |
| Ensemble of the document models and PromptORE + classifier | `doc-tune-ens+old` | 74.99 | 73.83 | 75.84 |

Everything was trained without a GPU (Mac M4 Pro). Still to do: label efficiency, PromptORE ablations, error analyses and the final test run (section 8 of the plan).
