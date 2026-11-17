# LLM representations of a pair in the PromptORE pipeline

**Status:** 2026-11-17. Dev results, seed 0, one run per row. No test run.

This note describes the LLM method of the project (M4 of [project_plan.md](project_plan.md)), as it
was finally run. It replaces the design of the open pull request #6: the LLM is not run end to
end, it replaces or completes the **representation of a pair** inside the PromptORE pipeline (M1
and M2). Everything else is the repository's code, so the gap with PromptORE measures the
representation only. All scores come from `scripts/evaluate.py`.

The code is in `notebooks/llm_pairs/`: `kgllm.py` (the library) and two Colab notebooks that embed
it (they need a T4 GPU and a DeepSeek API key, read from the Colab secret `DEEPSEEK_API_KEY`).
Both notebooks clone this repository at commit `5d3c818` and import `kg_extraction` from it.

## 1. What changes and what does not

Unchanged, called from `kg_extraction`: the candidate pairs and the type filter, the
`mention_sentences` context, the entity names, the train sample (all related pairs and 3
unrelated per related, seed 0, weighted), the logistic detector and the classifier per relation
(`RelationProbe`, M2), k-means and its naming by train labels (`ClusterTyper`, M1), the fact score
`P(related) x typing score`, the fact threshold and k chosen on dev.

Changed: the vector of a pair.

| Name | Vector of a pair |
|---|---|
| `bert` | PromptORE: the `[MASK]` state of `bert-base-uncased` (reference, recomputed in the same setting) |
| `qwen-state` | Last-layer state of `Qwen2.5-3B-Instruct` at the last token of the prompt below, where its answer starts. Centred on train, L2-normalised |
| `qwen-text` | Qwen answers the prompt, the answer is embedded by `all-mpnet-base-v2` |
| `api-text` | Same prompt answered by DeepSeek (`deepseek-chat`), same sentence encoder |
| `bert+api-text` | `bert` (centred, L2) concatenated with `api-text` |

The prompt gives the context and the two entity names and asks for one line,
`<relation> | <head type> | <tail type>`, where the relation is a free phrase of 1 to 5 words or
the word `none`. The model never sees the 96 relations and never sees a label. The vector that
is embedded is `head type | relation | tail type`.

## 2. Small scale: which representation (train_debug, dev_debug)

Fitted on `train_debug` (300 documents, 25,676 sampled pairs), scored on `dev_debug` (100
documents, 36,475 candidate pairs). F1 in %.

| Representation | M2 (classifier) | M1 (clusters) |
|---|---|---|
| `bert` (PromptORE) | 42.95 | 37.64 |
| `qwen-state` | 36.99 | 26.53 |
| `qwen-text` | 28.75 | 25.24 |
| `api-text` | 43.60 | 36.40 |
| `bert+api-text` | **52.06** | **44.60** |

- `bert+api-text` beats `bert` by 9.1 points with M2 (paired bootstrap, 95% interval 7.1 to
  11.2) and by 7.0 points with M1 (4.5 to 9.6).
- `api-text` alone equals `bert` (+0.65, interval -2.5 to +4.1). The two carry different
  information, which is why the concatenation helps.
- The 3B local model is clearly worse than BERT in both forms.

Two variants were tried with the DeepSeek answers and are negative results:

- **Clusters named by the LLM** instead of the train labels (DeepSeek chooses one of the 96
  relations from 8 pairs of the cluster): 22.8 F1 on the BERT clusters, against 37.6 with the
  train labels and 30.1 with one majority label per cluster. The LLM gives the majority label
  to about half of the clusters.
- **No label at all** (a pair is related when the answer is not `none`, k-means on dev, clusters
  named by the LLM): 17.0 F1 with DeepSeek and 9.5 with Qwen, the level of the type-pair
  baseline. DeepSeek answers `none` for 49% of the related pairs: detection is the bottleneck
  here as for every other method.

## 3. Larger scale: bert+api-text on the whole dev

Fitted on 1,000 train documents (`train_debug` and 700 others drawn with seed 1000; 89,484
sampled pairs, 22,371 related), scored on the whole dev (500 documents). The `bert` rows use
the same 1,000 documents.

| Run | F1 | Ign F1 | P | R | Detection F1 | Typing F1 | Macro F1 | Graph F1 |
|---|---|---|---|---|---|---|---|---|
| `t1k-bert+api-text-classifier` | **55.98** | 54.03 | 58.01 | 54.10 | 59.18 | 90.11 | 39.06 | 56.85 |
| `t1k-bert+api-text-clusters` | **51.86** | 49.88 | 54.22 | 49.70 | 57.81 | 84.71 | 29.21 | 52.52 |
| `t1k-bert-classifier` | 46.82 | 44.65 | 46.81 | 46.83 | 50.62 | 86.19 | 26.31 | 47.48 |
| `t1k-bert-clusters` | 43.16 | 41.17 | 43.72 | 42.61 | 49.16 | 81.50 | 18.67 | 43.88 |

- **Paired bootstrap on dev** (`scripts/compare_runs.py`): +9.17 F1 with M2 (95% interval 8.01
  to 10.30) and +8.70 with M1 (7.72 to 9.71).
- **Against the leaderboard:** with a third of the train documents, both rows are above
  `promptore-classifier` (49.22) and `promptore-clusters` (44.63), which use all 3,053. They
  stay below `atlop-frozen` (58.37) and the document models.
- **Where the gain is:** detection F1 goes from 50.6 to 59.2, the macro F1 from 26.3 to 39.1
  (rare relations 14.6 to 24.6), and the F1 on pairs that share no sentence from 43.2 to 53.1.

### The pair filter

To limit the API cost, only the dev pairs whose BERT detection probability (the detector of
`t1k-bert-classifier`) reaches 0.02 were sent to DeepSeek; the others have no fact. The
threshold was chosen on the 100 documents of `dev_debug`, which were all answered: it is the
largest that costs at most 0.5 F1 point there (56.75 without filter, 56.63 with it). It keeps
39% of the dev pairs; with the `dev_debug` documents, 94,958 of the 187,939 dev pairs were sent.
The `bert` rows are not filtered.

## 4. Limits

- One seed per row.
- 1,000 train documents, not 3,053: the rows are named `t1k-` for that reason. The comparison
  with the `t1k-bert` rows is exact; the comparison with the other leaderboard rows is at a
  disadvantage for these runs.
- The best k of M1 is the largest tried (400).
- The threshold, k and the pair filter were chosen on dev, as for the other methods. No test run.
- The DeepSeek answers depend on an external model that may change. The raw answers of every
  pair are kept with the run folders (`llm_answers_train.csv`, `llm_answers_dev.csv`).
- Cost: 62,513 API calls for section 2 and 124,343 more for section 3 (39.8 million input
  tokens in total).

## 5. Checks at merge time

- The notebooks pin commit `5d3c818`; `src/`, `scripts/evaluate.py` and `scripts/prepare_benchmark.py` have not changed
  since, so the candidates, the sample, the probes and the evaluator are those of `main`.
- The two `t1k-bert` rows were recomputed locally on the Mac (MPS) with the repository's code and the same 1,000
  documents: same sample (89,484 pairs, 22,371 related), dev F1 46.76 with the classifier (46.82 in Colab) and 42.83
  with clusters (43.16). The gap comes from GPU arithmetic and k-means, and is within the seed spread of
  `promptore-clusters` (0.45).
- The DeepSeek rows cannot be recomputed without the API answers, which are not versioned.

## 6. Reproduce

Open a notebook of `notebooks/llm_pairs/` in Colab with a T4 GPU and run all cells:

- `llm_small_scale_colab.ipynb`: section 2 (about 2 hours, mostly the local LLM).
- `llm_t1k_full_dev_colab.ipynb`: section 3. It reuses the answers of the first notebook from
  Google Drive (`MyDrive/kg_llm_cache/deepseek_answers.jsonl`).

Each notebook starts with a rehearsal on a few documents, writes the run folders and scores them
with `scripts/evaluate.py`. The run folders are not versioned, as for every other method.
