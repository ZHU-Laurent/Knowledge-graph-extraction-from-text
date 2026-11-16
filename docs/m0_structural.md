# M0': a structural-prior baseline with a detector, a typer and calibrated scores

**Status:** first working version, 2026-11-16. Dev results, seed 0.

This is a new comparison method. It reads **no text**, like the type-pair baseline it replaces, and
it is scored by the same shared evaluator as every other method
([collaboration.md](collaboration.md)). The code is `scripts/baseline_structural.py`; the run
folders are `results/m0-structural` (v1) and `results/m0-structural-v2` (v2).

---

## 1. What it is

The type-pair baseline M0 (`scripts/baseline_typepair.py`) estimates
`P(relation | head type, tail type, same sentence)` by counting on train, and predicts every
relation of a type pair whose probability passes one global threshold. Three limits follow from
that design:

1. **It cannot say "no relation".** Every type pair that carries a relation in train contributes
   its best relations to every pair of that type, so precision stays at the base rate.
2. **One threshold for the 96 relations.** A relation with a low prior can never be predicted,
   however related the pair is.
3. **No score.** The facts carry no confidence, so the graph metrics that need one
   (`graph_precision_at_N`, `graph_precision_support_*`) cannot be computed for M0 at all.

M0' keeps the promise of M0 -- only the structural metadata of the dataset is read, never the
text -- and fixes the three limits. It has **exactly the shape of PromptORE's M2**
(`relations/probe.py`): a logistic detector, then one logistic classifier per relation, and a
fact score of `P(related) x P(relation | related)`. The only difference is the representation of
a pair: structural features here, the `[MASK]` embedding of the prompt there.

That symmetry is the point. M0' and M2 differ in one thing, so the gap between them measures
what reading the text is worth.

### The features

All of them are structural. No token of the document is read.

| Block | Features |
|---|---|
| Types | head type (6), tail type (6) |
| Type pairs | head x tail (36), and head x tail x same sentence (36) |
| Distance | sentence distance between the closest mentions (7 buckets) |
| Mentions | number of mentions of the head and of the tail (2 x 5 buckets) |
| Links | has a Wikidata item (2) |
| Position | same type (1), size of the document (1) |
| v2 only | head x tail x sentence distance (252), subject position (2), first-mention sentence (1) |

106 features in v1, 361 in v2.

### The protocol

Identical to M2's, so that the comparison is clean:

- the train sample is the 67,808 related pairs and three times as many unrelated pairs, drawn
  with seed 0, with the weight that restores the true share of unrelated pairs;
- one fact threshold, chosen on dev;
- the fact score is `P(related) x P(relation | related)`, and it is written in the `score`
  column, so the graph metrics are computed.

---

## 2. Results (dev, seed 0, shared evaluator)

| | M0 | M0' v0 | M0' v1 | M0' v2 |
|---|---:|---:|---:|---:|
| Information used | M0's key | M0's key | + the rest | + interactions |
| **F1** | 18.75 | 19.59 | 21.64 | **23.05** |
| **Ign F1** | 17.69 | 17.98 | 20.02 | 21.52 |
| P | 17.42 | 13.72 | 16.06 | 17.73 |
| R | 20.30 | 34.29 | 33.16 | 32.95 |
| Detection F1 | 28.36 | 30.19 | 33.62 | 35.12 |
| Typing F1 | 61.29 | 61.38 | 61.02 | 61.11 |
| Macro F1 | 1.60 | 1.97 | 3.74 | **5.98** |
| Macro F1, frequent | 13.03 | | 18.32 | 19.70 |
| Macro F1, medium | 0.40 | | 2.90 | 6.04 |
| Macro F1, rare | 0.00 | | 0.33 | 1.15 |
| Graph F1 | 19.00 | 19.41 | 21.66 | 23.18 |
| Graph precision at 100 | n/a | 31.00 | 49.00 | **56.00** |
| Graph precision at 1000 | n/a | 26.90 | 38.10 | **51.80** |
| F1 95% interval | 17.72 to 19.77 | | 20.56 to 22.72 | 21.95 to 24.16 |

The three versions of the method differ only in the information they give the model:

- **v0** uses exactly what M0 uses (the joint type-pair key), written as one-hot features.
  It isolates the **model**.
- **v1** adds the other structural features: sentence distance, number of mentions, Wikidata
  links, same type, size of the document.
- **v2** adds the interactions between the type pair and the sentence distance, the position of
  the subject, and the first-mention sentence.

Paired bootstrap over the dev documents (`scripts/compare_runs.py`):

| Comparison | F1 gap | 95% interval | p |
|---|---:|---|---:|
| M0' v2 - M0 | **+4.30** | [+3.47, +5.12] | 0.000 |
| M0' v2 - M0' v1 | +1.41 | [+0.99, +1.85] | 0.000 |
| M0' v1 - M0' v0 | +2.05 | | |
| M0' v0 - M0 | +0.84 | [+0.28, +1.38] | 0.001 |

---

## 3. What the results say

### 3.1 The gain is 80% information and 20% model, and the two are complementary

The whole gain over M0 is +4.30 F1. The v0 run splits it, with the information held constant:

| | counting model (M0's) | probabilistic model (detector + typer) |
|---|---:|---:|
| **M0's information** (the type-pair key) | 18.75 (M0) | 19.59 (v0) |
| **the rest of the structure** | -- | 21.64 (v1), 23.05 (v2) |

- **the model is worth +0.84** (95% CI [+0.28, +1.38], p = 0.001)
- **the extra information is worth +3.46** (80% of the gain)
- **the interactions are worth +1.41** of that

The two are not additive. The model alone (v0) buys its recall by opening the threshold, and
precision falls from 17.42 to 13.72; it only recovers (17.73) once the extra information lets the
model tell apart the pairs of the same type. The detector raises recall from 20.3 to 34.3, the
information raises detection F1 from 30.2 to 35.1. Neither is enough alone: v0 is 3.5 points below
v2.

So the first correction to the obvious reading of the leaderboard: **M0's key throws away most of
what the dataset gives for free.** M0 keeps the type pair and the same-sentence flag and discards
the sentence distance, the number of mentions, the Wikidata links and the position of the
entities. Using them is worth more than the better model.

This matters for how the leaderboard is read: the gap between M0 and M2 is **not** only a gap
between "no text" and "text". A third of it would have appeared with more of the structure and
a better model.

### 3.2 Detection is the bottleneck even with no text

Typing F1 is 61.1 and detection F1 is 35.1. The same asymmetry as every other method of the
project, from a method that reads no text at all. The conclusion is therefore not an artefact of
the representation: **rejecting a pair is hard whatever the model sees.** The hypothesis of the
plan (section 1) holds for the cheapest method too, which makes it a property of the data.

### 3.3 The confidence ranks the graph, and M0' is the first baseline that shows it

M0' v2 predicts the 1,000 most confident edges of the corpus graph at **51.8% precision**, against
17.7% for all its edges. The confidence of a logistic model is informative enough to rank a
graph, even though it is not a probability of the truth. The two support levels are still worth
comparing: the edges stated by two documents are at 26.3% against 17.4% for the edges stated by
one.

M0 has no such column. This is the first row of the leaderboard with a confidence for the
baseline family.

### 3.4 Per-relation thresholds do not help (negative result)

One threshold per relation, each chosen to maximise the F1 of its own relation, gives **11.12**
dev F1 in sample and **12.41** cross-fitted, against **23.05** and **23.53** for one global
threshold. The reason is that maximising each relation's own F1 is not the same objective as
maximising the micro F1: a rare relation buys a little recall with a lot of false positives.

This independently reproduces the finding of [analysis_v2.md](analysis_v2.md) on the document
model ("Per-relation thresholds do not help... Dropped"), on a different method and a different
representation. The global threshold is kept.

For reference, the global threshold is not overfitted: cross-fitted on five folds of the dev
documents it gives 23.53 against 23.05 in sample.

### 3.5 What the text is worth, once the model is right

| | dev F1 | gain |
|---|---:|---|
| M0, type pair, counting, no text | 18.75 | |
| M0' v2, all the structure, calibrated, no text | 23.05 | +4.30 |
| M2, PromptORE + classifier, reads the text | 50.41 | +27.36 |
| M3 v2, document model, reads the document | 73.60 | +23.19 |

Of the 31.7 points that separate the type-pair baseline from PromptORE, **4.3 come from the
structural metadata and the model, and 27.4 require reading the text.** The structural prior is a
real but small part of PromptORE's score. This is the decomposition the leaderboard could not
give before.

### 3.6 Where the structural prior runs out

The most frequent relations are the ones the structure knows, and the failure is informative:

| Relation | gold | recall | F1 |
|---|---:|---:|---:|
| P131 located in the administrative territorial entity | 3846 | 0.716 | 0.286 |
| P17 country | 2739 | 0.364 | 0.223 |
| P27 citizenship | 823 | 0.756 | 0.354 |
| P150 contains administrative territorial entity | 662 | 0.072 | 0.086 |
| P361 part of | 444 | 0.038 | 0.057 |
| P710 participant of | 281 | 0.018 | 0.029 |

P131 and P27 are recoverable from the types alone (a place inside a place, a person inside a
country). P150, P361 and P710 need to know that the entities are part of the same larger entity,
which is written in the text and not in the structure: their recall is below 0.08.

---

## 4. Files and how to reproduce

```text
scripts/baseline_structural.py   the method
results/m0-structural-v0/         v0: M0's information only, M0''s model
results/m0-structural/            v1: 106 features, global threshold
results/m0-structural-v2/         v2: 361 features, global threshold
results/baseline-typepair/        M0, reproduced locally for the comparison
```

```bash
python scripts/prepare_benchmark.py
python scripts/baseline_typepair.py                                     # M0
python scripts/baseline_structural.py --run m0-structural-v0 --features v0 --thresholds global
python scripts/baseline_structural.py --run m0-structural   --features v1 --thresholds global
python scripts/baseline_structural.py --run m0-structural-v2 --features v2 --thresholds global
python scripts/evaluate.py results/m0-structural-v2
python scripts/compare_runs.py results/m0-structural-v2 results/baseline-typepair --subset dev
```

M0 was reproduced locally and gives dev_debug micro F1 0.185, the value recorded in
[collaboration.md](collaboration.md), so the two methods are compared on the same ground.

The run is deterministic (seed 0, no randomness in the model), and takes about three minutes on
a laptop CPU.

### What is not done

- **Test.** The run stays on dev, as the rules of the plan require until every choice is frozen.
  The method needs no new choice to move to test: `--test` predicts it with the dev threshold.
- **The leaderboard.** `report/leaderboard.csv` was left untouched. One command adds the row:
  `python scripts/evaluate.py results/m0-structural-v2 --leaderboard report/leaderboard.csv`.
- **Three seeds.** The method has no randomness beyond the negative sampling, so the seeds would
  only measure the sampling.
