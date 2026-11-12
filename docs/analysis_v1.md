# First review of the relation methods (v1)

**Date:** 2026-11-10. **Scope:** M0 to M3 of [project_plan.md](project_plan.md), entities known, Linked-Re-DocRED.
**Scores:** shared evaluator (`report/leaderboard.md`). **Error analysis:** `scripts/analyze_errors.py` on `dev`, seed 0. Test is not used for any analysis.

## 1. Results

Test scores are provisional (see the protocol in [evaluation.md](evaluation.md#reading-the-leaderboard)). Mean ± standard deviation over seeds 0, 1 and 2 (M0 is deterministic, one run). The interval is the 95% bootstrap interval over documents, averaged over seeds.

| Method | F1 (test) | 95% interval | Ign F1 | P | R | Detection F1 | Typing F1 | Macro F1 | Graph F1 |
|---|---|---|---|---|---|---|---|---|---|
| M3, ATLOP-style, frozen encoder | **57.70 ± 0.10** | 55.8 to 59.7 | **56.76 ± 0.18** | 74.35 | 47.18 | 60.14 | 92.17 | 36.72 | 59.20 |
| M2, PromptORE + classifier | **50.41 ± 0.14** | 48.6 to 52.2 | **48.07 ± 0.11** | 49.10 | 51.79 | 53.21 | 88.91 | 30.58 | 50.70 |
| M1, PromptORE + clusters | **45.93 ± 0.38** | 44.0 to 47.7 | **43.56 ± 0.48** | 44.95 | 47.02 | 51.29 | 83.19 | 20.97 | 46.01 |
| M0, type pairs | **18.86** | 17.9 to 19.9 | **17.80** | 17.45 | 20.52 | 28.14 | 61.98 | 1.71 | 19.42 |

Paired bootstrap on dev, 3 seeds pooled (`scripts/compare_runs.py`): M3 beats M2 by 9.2 points (95% interval 7.8 to 10.4), and M2 beats M1 by 4.6 points (4.1 to 5.2). Both gaps are far beyond document sampling noise and seed variance (section 5).

## 2. What the errors look like (dev)

### Detection is the bottleneck, not typing

| | M3 | M2 | M1 |
|---|---|---|---|
| Missed facts whose pair is never predicted | 7,827 of 8,781 (89%) | 7,756 of 8,746 (89%) | 8,478 of 9,921 (85%) |
| False facts on a pair with no gold relation | 2,523 of 3,158 (80%) | 7,580 of 8,859 (86%) | 6,996 of 8,709 (80%) |
| Typing F1 | 91.5 | 88.3 | 82.4 |

Once a pair is known to be related, every method picks the right relation about 9 times out of 10. Almost all the loss comes from deciding which of the about 390 pairs of a document are related (7% of pairs are).

### M3 is too cautious

M3 predicts 11,661 facts for 17,284 gold facts (precision 72.9, recall 49.2). Its learned threshold is too high for F1: facts scored just above it are still right 46% of the time. Lowering the threshold on dev:

| Threshold | 0.50 (learned) | 0.40 | 0.30 | 0.25 | 0.20 | 0.10 |
|---|---|---|---|---|---|---|
| Dev F1 | 58.75 | 59.93 | 60.23 | **60.32** | 60.08 | 58.23 |

A cheap +1.6, but a small one: even at threshold 0.05 recall only reaches 71.5. The ranking itself is limited.

### Rare relations are lost

Recall by train frequency of the relation (10 frequent relations, medium, rare = fewer than 200 train facts):

| | Frequent | Medium | Rare |
|---|---|---|---|
| M3 | 58.1 | 38.3 | 18.0 |
| M2 | 60.1 | 35.2 | 18.1 |
| M1 | 56.8 | 22.9 | **6.7** |

Macro F1 (37.5 for M3) is far below micro F1 for every method. M1 is the worst on rare relations: a cluster is named after the relations of its train pairs, and a rare relation rarely dominates a cluster.

### Distance between the entities hurts PromptORE more than M3

| Sentence distance | 0 | 1 | 2 | 3+ |
|---|---|---|---|---|
| M3 recall / precision | 51.9 / 74.8 | 48.5 / 70.0 | 45.0 / 70.3 | 46.0 / 73.2 |
| M2 recall / precision | 50.9 / 57.7 | 43.5 / 49.3 | 47.9 / 42.3 | 53.8 / 39.0 |

M3 encodes the whole document, so distance changes little. M2 sees only the sentences of the mentions in its prompt, and its precision drops from 58 to 39 on distant pairs.

### Other breakdowns (M3)

- **Entity types:** recall 61 on LOC-LOC, 55 on ORG-LOC, but 31 on PER-PER and 32 to 34 on MISC pairs. M1 falls to 4 on PER-PER (family relations are rare and get no cluster).
- **Long documents:** recall 55 with fewer than 15 entities, 44 with 25 or more.
- **Most missed relations:** P131 (located in), P17 (country), P150 (contains), P527 (has part), P361 (part of). These are geographic and part-of hierarchies, often inferred by the annotators rather than written.
- **Inference rules do not help:** adding the inverse facts (P131 to P150, P361 to P527...) and the country implied by "located in" lowers dev F1 by about 4 points for every method (M3: 58.75 to 54.98). Re-DocRED does not annotate every inferred fact, so such rules add as many false facts as true ones.
- **Annotation gaps explain few false facts:** only 50 of M3's 3,158 false facts are the inverse of a gold fact, and 113 are gold in another document. Precision is not much underestimated.
- **Calibration:** for M1 and M2, precision rises steadily with the score (28% at 0.1 to 0.2, 96% above 0.9), so scores can rank graph edges. For M3, facts scored above 0.9 are right 87% of the time.

### The methods are complementary

M3 and M2 share 5,967 correct facts, but each finds about 2,500 that the other misses. The union of their correct facts covers 64.1% of the gold facts, against 49.2% for M3 alone.

## 3. Where the weaknesses come from

| Weakness | Main cause |
|---|---|
| Low detection recall (M3) | The encoder is **frozen**: BERT was never adapted to relations, and only a small head is trained. Published ATLOP fine-tunes a RoBERTa-large encoder and reaches 77.6 F1 (Ign F1 76.8) on Re-DocRED test, 20 points above our frozen BERT-base. Part of the gap is the larger encoder, part is fine-tuning |
| Under-prediction (M3) | The adaptive threshold learned in training is too strict for F1 |
| Many false pairs (M2, M1) | The PromptORE detector sees one prompt per pair with only the mention sentences, through a frozen masked language model; it cannot use the rest of the document |
| Rare relations | Few examples per relation, no reweighting of the loss; for M1, clusters named by majority vote |
| PER-PER, MISC pairs, long documents | Same causes: little data per relation, and more candidate pairs per document |

## 4. Plan to fix them

Ordered by expected gain over cost. Each step is chosen on `dev` and scored with the shared evaluator, 3 seeds.

| # | Step | Fixes | Expected effect | Cost |
|---|---|---|---|---|
| 1 | Tune the threshold of M3 on dev (global offset, then one per relation) | Under-prediction | +1.6 measured with one global threshold | Minutes |
| 2 | Fine-tune the encoder of M3 (bert-base-cased, then roberta-large), ATLOP settings | Low detection recall | The largest gain: the published RoBERTa-large version is 20 points above ours | GPU, hours per run |
| 3 | Combine M3 and M2: average their scores per fact, weight and threshold chosen on dev | Complementary errors | Union recall is 64 against 49; gain to measure | Minutes |
| 4 | Rare relations: adaptive focal loss or class-balanced weights in M3, and report macro F1 by bucket | Long tail | Higher macro F1 and rare recall | One training per variant |
| 5 | PromptORE with M3 as detector: M3 decides which pairs are related, PromptORE (clusters or classifier) types them | Weak PromptORE detection | Isolates PromptORE typing, the research question; M2 typing F1 is already 89 | Minutes, once M3 exists |
| 6 | M1: name clusters with soft assignments (several relations per cluster), larger k, and export clusters for B³ / V / ARI | Rare relations in M1 | Better rare recall; clustering metrics comparable with the paper | Minutes |
| 7 | Experiment B (labels needed) and the remaining analyses of experiment D | Research questions | | After 1 to 6 |
| 8 | Final test run: one per method, `test_status: final` in its card | Provisional test | Final table | Once |

Dropped: hand-written inference rules (inverse and transitive facts), which lower F1 by about 4 points (section 2).

## 5. Seed variance

Seeds 0, 1 and 2. For M3 the seed sets the initialization and the order of the batches; for M2 and M1 it draws the unrelated train pairs (and k-means for M1). Settings chosen on dev for each seed: M2 threshold 0.18 every time; M1 k = 400 every time, threshold 0.15 to 0.17; M3 best epoch 19, 15 and 21.

| Method | Dev F1 by seed | Test F1, mean ± std |
|---|---|---|
| M3 | 58.75, 57.81, 58.55 | 57.70 ± 0.10 |
| M2 | 49.24, 49.23, 49.20 | 50.41 ± 0.14 |
| M1 | 44.15, 45.03, 44.71 | 45.93 ± 0.38 |

The standard deviation over seeds (0.1 to 0.5 point) is smaller than the bootstrap interval over documents (about ± 2 points), and much smaller than the gaps between methods. The precision and recall of M3 move more (P 74.4 ± 2.7): the seeds change where the learned threshold falls, which is one more reason to tune it on dev (plan step 1).
