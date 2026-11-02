# Knowledge graph extraction from text

Text mining project: build a knowledge graph from raw text with a pipeline that needs no annotations from the target domain.

The relation step uses [PromptORE](https://arxiv.org/abs/2304.01209), an unsupervised method that groups entity pairs by relation but does not detect entities or name relations. We add the missing stages and measure how much quality each one loses.

## Pipeline

```text
Raw text
  → sentences
  → entities (GLiNER)
  → candidate entity pairs
  → filter pairs without a relation (added after the first version)
  → relation embeddings and clusters (PromptORE)
  → filter results
  → name and merge clusters (LLM)
  → triples
  → normalized entities and knowledge graph
```

See [docs/project_plan.md](docs/project_plan.md) for the design decisions, evaluation and work plan, and [docs/personal_researches.md](docs/personal_researches.md) for the dataset selection.

## Project structure

```text
├── configs/              Experiment settings
├── data/
│   ├── raw/              Downloaded datasets (not versioned)
│   └── processed/        Converted datasets (not versioned)
├── docs/                 Subject, plan and dataset selection
├── external/             Original PromptORE code (not versioned)
├── notebooks/            Exploration and error analysis
├── report/               Final report and figures
├── results/              Metrics, predictions and graphs (not versioned)
├── scripts/              Command-line entry points
├── src/kg_extraction/
│   ├── data/             Dataset loading and conversion
│   ├── entities/         Sentence splitting and entity detection
│   ├── pairs/            Candidate pairs and relation filter
│   ├── relations/        Relation embeddings, clustering and result filter
│   ├── naming/           Cluster naming and merging
│   ├── graph/            Triples, entity normalization and graph construction
│   └── evaluation/       Metrics for each stage
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
| Reproduce the baseline | [FewRel](https://github.com/thunlp/FewRel) |
| Evaluate the pipeline | [SciERC](http://nlp.cs.washington.edu/sciIE/) |
| Test versatility, if time allows | [FRAME](https://doi.org/10.5281/zenodo.18724632) |
| Test longer documents, optional | [SciER](https://github.com/edzq/SciER) |

## Status

Project structure and plan only: nothing is implemented yet.
