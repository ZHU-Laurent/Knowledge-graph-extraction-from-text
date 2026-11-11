# Evaluating a method on Linked-Re-DocRED

There are two ways to score a method. Both use the same subsets (fixed with seed 0) and the same metric code, so
their numbers are directly comparable.

| | Script version | Notebook cell version |
|---|---|---|
| File | `scripts/evaluate.py` | `notebooks/redocred_eval_cell.py` |
| Input | a run folder of CSV prediction files | a Python list of tuples or a DataFrame |
| Output | `scores.json`, per-relation CSV, a row in `report/leaderboard.csv` | a dict of scores, printed summary |
| Use it for | numbers that go in the report | quick iteration inside a notebook (Colab, Jupyter) |

Iterate with the cell, then write the files and run the script for anything that goes in the report.
The rules (subsets, supervision levels, dev for tuning, test scored once) are in [collaboration.md](collaboration.md).

## The task in one paragraph

Each document comes with its entities already given (types and mentions). A method looks at every ordered entity
pair `(h, t)` of a document and outputs zero, one or several relations `r` (Wikidata codes such as `P17` "country").
A prediction is a fact `(doc_id, h, r, t)`. A pair with no predicted fact means "no relation". About 6% of ordered
pairs are related, so predicting too many facts costs precision quickly.

## Subsets

| Subset | Documents | Ordered pairs | Use |
|---|---|---|---|
| `train_debug` | 300 random train docs | 112,872 | fit things quickly |
| `dev_debug` | 100 random dev docs | 37,668 | quick tuning and debugging |
| `train` | 3,053 | 1,193,092 | full training data |
| `dev` | 500 | 193,232 | reported dev numbers |
| `test` | 500 | 198,670 | final evaluation, once |

`dev_debug` gives almost the same scores as `dev` (type-pair baseline: micro F1 0.185 vs 0.188).

## Option 1: notebook cell

Paste the whole content of [`notebooks/redocred_eval_cell.py`](../notebooks/redocred_eval_cell.py) into one cell
and run it. It needs only `numpy`, `pandas` and `scikit-learn`, and downloads the three Linked-Re-DocRED files into
`data/raw/Linked-Re-DocRED/` the first time (about 100 MB).

```python
docs  = load_subset("dev_debug")        # list of documents: doc_id, title, sents, entities (no relations)
pairs = candidate_pairs("dev_debug")    # DataFrame: doc_id, h, t, h_type, t_type, same_sentence

# ... your method ...
predictions = [("1234", 0, "P17", 3), ("1234", 2, "P131", 0)]   # (doc_id, h, r, t), or a DataFrame

scores = evaluate_redocred(predictions, "dev_debug")
scores["micro_f1"], scores["per_relation"].head()
```

What each helper gives:

| Function | Returns |
|---|---|
| `load_subset(subset)` | documents without relations. Each entity has `id` (its index, used as `h`/`t`), `type`, `name` (longest mention) and `mentions` (`name`, `sent_id`, `pos`) |
| `candidate_pairs(subset)` | every ordered pair `h != t` with entity types and whether they share a sentence |
| `gold_facts("train")` | the set of train facts `(doc_id, h, r, t)`, for methods allowed to use labels |
| `evaluate_redocred(predictions, subset, clusters=None)` | the scores below as a dict, plus `per_relation` (a DataFrame) |

With clusters (one cluster id per pair, any hashable value), the clustering metrics are added:

```python
cl = pd.DataFrame({"doc_id": ..., "h": ..., "t": ..., "cluster": ...})
scores = evaluate_redocred(predictions, "dev_debug", clusters=cl)
```

A complete example, the type-pair baseline (`scripts/baseline_typepair.py`) rewritten for the cell:

```python
from collections import Counter, defaultdict
train = candidate_pairs("train_debug")
gold = defaultdict(set)
for d, h, r, t in gold_facts("train_debug"):
    gold[(d, h, t)].add(r)
n_key, n_key_rel = Counter(), defaultdict(Counter)
for row in train.itertuples():
    key = (row.h_type, row.t_type, row.same_sentence)
    n_key[key] += 1
    for r in gold.get((row.doc_id, row.h, row.t), ()):
        n_key_rel[key][r] += 1

preds = [(p.doc_id, p.h, r, p.t) for p in candidate_pairs("dev_debug").itertuples()
         for r, c in n_key_rel[(p.h_type, p.t_type, p.same_sentence)].items()
         if c / n_key[(p.h_type, p.t_type, p.same_sentence)] >= 0.1]
evaluate_redocred(preds, "dev_debug");   # micro F1 0.187 (fitted on train_debug only)
```

## Option 2: script and leaderboard

```bash
python scripts/prepare_benchmark.py          # once: builds data/processed/benchmark/
# your method writes results/<run>/card.json and results/<run>/redocred_<dev_debug|dev|test>_facts.csv
python scripts/evaluate.py results/<run> --leaderboard report/leaderboard.csv
```

`redocred_<subset>_facts.csv` has the columns `doc_id,h,r,t`; an optional `redocred_<subset>_clusters.csv` has
`doc_id,h,t,cluster`. The file formats and the `card.json` fields are described in
[collaboration.md](collaboration.md#prediction-files). `scripts/baseline_typepair.py` is a worked example.

From a notebook, the predictions above can be written with:

```python
import os, json
os.makedirs("results/my-run", exist_ok=True)
pd.DataFrame(preds, columns=["doc_id", "h", "r", "t"]).to_csv("results/my-run/redocred_dev_facts.csv", index=False)
json.dump({"run": "my-run", "author": "...", "method": "...", "supervision": "supervised",
           "training_data": "...", "models": "...", "description": "..."}, open("results/my-run/card.json", "w"))
```

## Reading the leaderboard

`report/leaderboard.csv` has every metric. `report/leaderboard.md`, written at the same time, shows only the metrics
below, with one table per subset and the best F1 first. In the CSV the same metrics come first, in this order.

| Tier | Metric | Question it answers |
|---|---|---|
| **1. Key, rank on these** | `micro_f1` | How many of the relations to find are found, and found right? Main score |
| | `ign_f1` | Same, without the facts already seen in train. Guards against memorizing, and is the score published DocRED papers report next to F1 |
| 2. Diagnosis | `micro_p`, `micro_r` | Are the errors false facts or missed facts? |
| | `detect_f1` | Does the method find which pairs are related? |
| | `typing_f1` | On related pairs it found, does it choose the right relation? |
| | `macro_f1` | Does it work on every relation, rare ones included? |
| 3. Analysis | the other columns | Evidence-only facts, intra / inter sentence, frequent / medium / rare relations, clustering |

Read tier 1 to compare methods, then tier 2 to see why one method is ahead. Make every choice on `dev` (or
`dev_debug` while iterating) and report `test` once per method, with frozen settings.

## Metrics

| Key | Meaning |
|---|---|
| `micro_p`, `micro_r`, `micro_f1` | facts `(doc, h, r, t)` against all gold facts. **Main score** |
| `ign_p`, `ign_f1` | Ign F1 of DocRED: correct facts whose head mention name, tail mention name and relation already form a train fact are removed from the hits and from the predictions. Recall is unchanged |
| `typing_p`, `typing_r`, `typing_f1` | only the pairs related in gold and predicted related: is the relation right? Missed and false pairs are detection errors, not typing errors |
| `f1_intra_sentence`, `f1_inter_sentence` | F1 on pairs that share a sentence / do not |
| `micro_f1_ci_low`, `micro_f1_ci_high` | 95% interval of `micro_f1`, from 1,000 resamples of the documents (same draws in both versions) |
| `graph_p`, `graph_r`, `graph_f1`, `graph_precision_*` | corpus graph metrics, **script only** (they reuse `src/kg_extraction/graph`). See [collaboration.md](collaboration.md#what-the-evaluator-reports-for-linked-re-docred) |
| `macro_f1` | F1 averaged over the relation types present in the gold of the subset |
| `macro_f1_frequent / _medium / _rare` | macro F1 on the 10 most frequent train relations / the others / relations with fewer than 200 train facts |
| `evid_p`, `evid_r`, `evid_f1` | only gold facts annotated with evidence sentences. Facts without evidence (often inferred, not written) count neither as hits nor as errors. Closest to "explicit relations" |
| `recall_intra_sentence`, `recall_inter_sentence` | recall on pairs that share a sentence / do not |
| `detect_p`, `detect_r`, `detect_f1` | is the pair related at all, relation type ignored |
| `cl_b3_f1`, `cl_v_f1`, `cl_ari`, ... | clustering of related pairs with a single gold label, only if clusters are given |
| `n_gold`, `n_pred`, `n_tp` | counts (cell version) |

Predictions with `r = "no_relation"` are ignored. A method may skip pairs (for example by type), but it is always
scored against all gold facts of the subset, so a skipped real relation costs recall.

## Reference scores

Type-pair baseline (reads no text):

| Subset | micro P | micro R | micro F1 | macro F1 | evid F1 | detect F1 | cl B³ F1 | cl V F1 | cl ARI |
|---|---|---|---|---|---|---|---|---|---|
| dev_debug | 0.168 | 0.205 | 0.185 | 0.016 | 0.117 | 0.286 | 0.358 | 0.620 | 0.288 |
| dev | 0.174 | 0.203 | 0.188 | 0.016 | 0.133 | 0.284 | 0.349 | 0.603 | 0.294 |

## The two versions stay in sync

`tests/test_evaluation.py::test_eval_cell_matches_script` runs both versions on the same toy data and checks that
every shared metric is equal. If you change the metrics, change both files and run `pytest tests/`.
