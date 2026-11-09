# Working together: one benchmark, one evaluator

Each of us can try a different method, as long as every method is scored on the same data, with the same
code and the same rules. Methods then only differ by their **prediction files**.

## The rule in one sentence

Your method writes prediction files in the formats below. `scripts/evaluate.py` scores them and adds a row to
`report/leaderboard.csv`. Nobody computes their own metrics for the report.

## Two tracks

| Track | Data | What a method predicts | Metrics | Role |
|---|---|---|---|---|
| **A. Clustering** | FewRel NYT (2,500 instances, 25 relations), FewRel PubMed (1,000, 10) | a cluster id for each instance | B³, V-measure, ARI | reproduce PromptORE (plan step 1), compare relation representations (BERT, LLM, …) |
| **B. Main** | Linked-Re-DocRED dev / test | facts `(doc_id, h, r, t)`; no row means "no relation" | micro P/R/F1 and macro F1 on facts, plus B³/V/ARI on single-label related pairs if clusters are given | the project's research question |

## Setup (once)

```bash
python scripts/prepare_benchmark.py        # downloads FewRel NYT/PubMed, writes data/processed/benchmark/
# Linked-Re-DocRED: copy dev_revised.json and test_revised.json into data/raw/Linked-Re-DocRED/
```

`data/processed/benchmark/{nyt,pubmed}.jsonl` holds the inputs with stable ids (`nyt-00000`, …): tokens and the
head/tail spans, **without labels**. Methods read only these files. The `*_gold.csv` files are read only by the
evaluator.

## Prediction files

Put everything for one run in `results/<run>/`. The folder is not versioned.

| File | Columns | Notes |
|---|---|---|
| `card.json` | see below | required |
| `fewrel_<nyt\|pubmed>_<setting>_seed<N>.csv` | `id,cluster` | one row per benchmark id. `setting` is `k_true` (k = 25 / 10 is given) or `k_est` (k estimated without labels); other names are allowed for extra variants |
| `redocred_<dev\|test>_facts.csv` | `doc_id,h,r,t` | `doc_id` = the document's `id` field, `h`/`t` = entity indexes, `r` = Wikidata code (P17, …). A pair may have several rows |
| `redocred_<dev\|test>_clusters.csv` | `doc_id,h,t,cluster` | optional, for B³/V/ARI |

`card.json` describes the run, so the leaderboard says how each number was obtained:

```json
{
  "run": "promptore-bert-v1",
  "author": "Laurent",
  "method": "PromptORE, BERT [MASK] embedding + k-means",
  "supervision": "none",
  "training_data": "none",
  "models": "bert-base-uncased",
  "description": "template '{sent} {e1} [MASK] {e2}', centred + L2-normalised, 5 seeds"
}
```

`supervision` must be one of:

- `none`: no label is used anywhere (the PromptORE setting)
- `labels_for_mapping`: clusters are unsupervised, but train labels map clusters to relations (our main pipeline)
- `supervised`: a model is trained on labels (linear classifier, ATLOP, fine-tuned LLM)

Numbers from different supervision levels go in separate rows of the report's tables and are never ranked together.

## Scoring

```bash
python scripts/evaluate.py results/<run> --leaderboard report/leaderboard.csv
```

The script checks that every benchmark id is predicted exactly once, scores each file, averages over seeds
(mean ± std), writes `results/<run>/scores.json` and replaces that run's rows in the leaderboard.
Commit `report/leaderboard.csv` together with the code of the method.

## Shared rules

1. **Same inputs.** Use the given entity spans and the official splits. Do not edit or filter the benchmark files.
2. **Labels.** Track A methods never read `*_gold.csv`. In track B, train labels are allowed for `labels_for_mapping` and `supervised` runs. Every decision (k, thresholds, prompts) is made on **dev**. **Test** is scored once, at the end, by all of us together.
3. **Seeds.** Anything random (k-means, sampling) runs with seeds 0 to 4. The evaluator reports mean ± std.
4. **k.** Report `k_true` and `k_est` separately. `k_est` must be chosen without labels.
5. **LLMs and APIs.** Record the exact model name and date in `card.json`. Use temperature 0. Keep the raw answers, since API models change over time.
6. **Reproducibility.** Every row of the leaderboard points to code in this repository (a script or notebook) that regenerates its prediction files.

## Git workflow

- One branch per method, e.g. `method/llm-embeddings`, merged through a pull request.
- Shared code (`src/kg_extraction/data`, `src/kg_extraction/evaluation`, the benchmark format) only changes in its own pull request, reviewed by another member, because it affects everyone's scores.
- Data, models and `results/` stay out of git, as `.gitignore` already enforces.
