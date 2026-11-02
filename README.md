# Knowledge graph extraction from text

Text mining project: extract entities and explicit relations from text, then build a knowledge graph.

Our baseline is [PromptORE](https://arxiv.org/abs/2304.01209), an unsupervised relation extraction method, which we first reproduce on its FewRel dataset. We then compare three relation extractors (rules, a fine-tuned Transformer and an LLM) to answer one question: which approach produces the best graph, at what cost, and with what kinds of errors?

See the full plan in [docs/project_plan.md](docs/project_plan.md) and the dataset selection in [docs/personal_researches.md](docs/personal_researches.md).

## Project structure

```text
├── configs/              Experiment settings
├── data/
│   ├── raw/              Downloaded datasets (not versioned)
│   └── processed/        Converted datasets (not versioned)
├── docs/                 Subject, plan and research notes
├── notebooks/            Exploration and error analysis
├── report/               Final report and figures
├── results/              Metrics, predictions and graphs (not versioned)
├── scripts/              Command-line entry points
├── src/kg_extraction/
│   ├── data/             Dataset loading and conversion
│   ├── entities/         Entity detection and classification
│   ├── relations/        Rule-based, Transformer and LLM extractors
│   ├── graph/            Entity normalization and graph construction
│   └── evaluation/       Precision, recall, F1 and error analysis
└── tests/                Unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

Data files are not stored in this repository. Download them into `data/raw/`:

| Stage | Dataset |
|---|---|
| Baseline | [FewRel](https://github.com/thunlp/FewRel) |
| Quick test | [CoNLL04](https://huggingface.co/datasets/DFKI-SLT/conll04) |
| Final version | [SciERC](http://nlp.cs.washington.edu/sciIE/) |
| Optional extension | [SciER](https://github.com/edzq/SciER) |
| Optional extension | [FRAME](https://doi.org/10.5281/zenodo.18724632) |

## Status

Project structure only: nothing is implemented yet.
