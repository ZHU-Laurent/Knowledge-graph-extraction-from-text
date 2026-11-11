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
| **B. Main** | Linked-Re-DocRED, fixed subsets (below) | facts `(doc_id, h, r, t)`; no row means "no relation" | micro P/R/F1 and macro F1 on facts, with breakdowns (below) | the project's research question |

## Setup (once)

```bash
# 1. download train_revised.json, dev_revised.json, test_revised.json from
#    https://github.com/alteca/Linked-DocRED/tree/main/Linked-Re-DocRED into data/raw/Linked-Re-DocRED/
# 2. build every benchmark file (FewRel is downloaded automatically)
python scripts/prepare_benchmark.py
```

Everything lands in `data/processed/benchmark/`:

| File | Content |
|---|---|
| `{nyt,pubmed}.jsonl` | FewRel inputs with stable ids (`nyt-00000`, …): tokens and head/tail spans, **no labels** |
| `{nyt,pubmed}_gold.csv` | FewRel labels, read only by the evaluator |
| `redocred_<subset>_docs.jsonl` | one document per line: `doc_id`, `title`, `sents`, `entities` (type, longest-mention `name`, mentions, Wikidata id), **no relations** |
| `redocred_<subset>_pairs.csv` | every ordered entity pair: `doc_id,h,t,h_type,t_type,same_sentence` |
| `redocred_subsets.json` | document ids of each subset |

### Linked-Re-DocRED subsets (fixed with seed 0, never re-drawn)

| Subset | Documents | Ordered pairs | Use |
|---|---|---|---|
| `train_debug` | 300 random train docs | 112,872 | fit things quickly (k-means, mappings) |
| `dev_debug` | 100 random dev docs | 37,668 | quick tuning and debugging |
| `train` | 3,053 | 1,193,092 | full training data |
| `dev` | 500 | 193,232 | all reported dev numbers |
| `test` | 500 | 198,670 | final evaluation, once |

Labels: train labels may be used by `labels_for_mapping` and `supervised` runs, dev labels for tuning, test
labels only by the evaluator. Methods may filter the candidate pairs themselves (types, sentence distance),
but are always scored against **all** gold facts of the subset, so a filter that drops real relations costs recall.

## Prediction files

Put everything for one run in `results/<run>/`. The folder is not versioned.

| File | Columns | Notes |
|---|---|---|
| `card.json` | see below | required |
| `fewrel_<nyt\|pubmed>_<setting>_seed<N>.csv` | `id,cluster` | one row per benchmark id. `setting` is `k_true` (k = 25 / 10 is given) or `k_est` (k estimated without labels); other names are allowed for extra variants |
| `redocred_<dev_debug\|dev\|test>_facts[_seed<N>].csv` | `doc_id,h,r,t[,score]` | `doc_id` and `h`/`t` as in the pairs file, `r` = Wikidata code (P17, …). A pair may have several rows; pairs without a row are "no relation". One file per seed with `_seed<N>`; the evaluator reports mean ± std. The optional `score` (confidence in [0, 1]) ranks the graph edges for precision at N |
| `redocred_<dev_debug\|dev\|test>_clusters[_seed<N>].csv` | `doc_id,h,t,cluster` | optional, for B³/V/ARI |

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

Optional: `"test_status": "final"` marks the single test run of a method, made after every choice on dev. Without
it, test rows are shown as `provisional` in the leaderboard.

## What the evaluator reports for Linked-Re-DocRED

| Column | Meaning |
|---|---|
| `micro_p`, `micro_r`, `micro_f1` | facts `(doc, h, r, t)`, all gold facts. Main score |
| `ign_p`, `ign_f1` | Ign F1 of DocRED: hits already seen as a train fact (same mention names and relation) are not counted. Second key score |
| `typing_p`, `typing_r`, `typing_f1` | on pairs related in gold and predicted related, is the relation right? |
| `f1_intra_sentence`, `f1_inter_sentence` | F1 on pairs that share a sentence / do not |
| `micro_f1_ci_low`, `micro_f1_ci_high` | 95% bootstrap interval of `micro_f1`, resampling documents |
| `graph_p`, `graph_r`, `graph_f1` | facts merged into one corpus graph (entities merged on their Wikidata item, dates and numbers on their text), against the gold graph built the same way. Script only |
| `graph_precision_at_<N>`, `graph_precision_support_<1\|2\|3+>` | precision of the N most confident edges (needs a `score` column), and of edges stated in 1, 2 or 3+ documents. Script only |
| `macro_f1` | F1 averaged over the gold relation types |
| `macro_f1_frequent / _medium / _rare` | macro F1 on the 10 most frequent train relations / the rest / relations with fewer than 200 train facts |
| `evid_p`, `evid_r`, `evid_f1` | only the gold facts annotated with evidence sentences (43% of dev). Re-DocRED added many facts without evidence, often inferred rather than written; predicting those is neither rewarded nor penalised here. Closest to "explicit relations" in the subject, but the evidence field is itself incomplete |
| `recall_intra_sentence`, `recall_inter_sentence` | recall on pairs that share a sentence / do not |
| `detect_p`, `detect_r`, `detect_f1` | is the pair related at all? (relation type ignored) |
| `cl_b3_f1`, `cl_v_f1`, `cl_ari`, … | clustering metrics on related pairs with a single gold label, if a clusters file is given |

`results/<run>/redocred_<subset>_per_relation.csv` gives P/R/F1 for each relation. Which metrics to look at
first: [evaluation.md](evaluation.md#reading-the-leaderboard).

### Reference: `scripts/baseline_typepair.py`

A baseline that reads no text: each pair gets the relations whose probability given (head type, tail type,
same sentence) passes a threshold tuned on dev_debug. It is also a worked example of the file formats.

| Subset | micro P | micro R | micro F1 | macro F1 | evid F1 | detect F1 | cl B³ F1 | cl V F1 | cl ARI |
|---|---|---|---|---|---|---|---|---|---|
| dev_debug | 0.168 | 0.205 | 0.185 | 0.016 | 0.117 | 0.286 | 0.358 | 0.620 | 0.288 |
| dev | 0.174 | 0.203 | 0.188 | 0.016 | 0.133 | 0.284 | 0.349 | 0.603 | 0.294 |

Any method that reads the text should beat it. The two subsets give almost the same scores, so dev_debug is a
reasonable proxy while iterating.

## Scoring

```bash
python scripts/evaluate.py results/<run> --leaderboard report/leaderboard.csv
```

For a version to paste into a notebook, see [evaluation.md](evaluation.md).

The script checks that every benchmark id is predicted exactly once, scores each file, averages over seeds
(mean ± std), writes `results/<run>/scores.json` and replaces that run's rows in the leaderboard.
Commit `report/leaderboard.csv` and `report/leaderboard.md` together with the code of the method.

Each Linked-Re-DocRED row also has a 95% interval of F1 (`micro_f1_ci_low`, `micro_f1_ci_high`), from 1,000
resamples of the documents. To check that one method really beats another, on the same documents:

```bash
python scripts/compare_runs.py results/<run A> results/<run B> --subset dev
```

It prints the F1 gap, its 95% interval and a p-value (paired bootstrap). For an error analysis on dev
(detection or typing errors, distance, entity types, rare relations, calibration):
`python scripts/analyze_errors.py results/<run> ... --subset dev`.

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
