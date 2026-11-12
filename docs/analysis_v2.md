# Second round: what each change brings (v2)

This round follows the plan of [analysis_v1.md](analysis_v1.md). No GPU was used (Mac M4 Pro, MPS). In sections 1 to 5 every encoder stays frozen: its token states are cached once and only what sits on top is trained, in about 1 to 2 minutes per epoch. Section 6 fine-tunes the last layers of the encoder, which brings the largest gain of the round. Every number comes from the shared evaluator (`scripts/evaluate.py`).

## 1. The new base method

M3 scored pairs from pre-computed pair vectors. The new model (`relations/doc_model.py`, `scripts/train_doc_model.py`) computes the same ATLOP head document by document, so that trainable parts can be added between the frozen encoder and the head:

- **token layers:** Transformer layers over the frozen token states;
- **pair features:** entity types, number of mentions and sentence distance of the pair, as learned embeddings;
- **pair layers:** reasoning over the entity-pair grid of the document, where pair (i, j) is updated from the pairs (i, k) and (k, j), the pairs with the same head or tail, and the pair (j, i).

The fact threshold is one global score threshold chosen on dev, instead of the threshold learned in training.

## 2. Gain of each change (dev, seed 0)

`bert-base-cased` frozen, 20 epochs, learning rate 3e-4 unless stated.

| Run | Change | F1 | Gain | Ign F1 | Macro F1 | F1 cross-fitted |
|---|---|---|---|---|---|---|
| `atlop-frozen` | M3 of v1 (3 seeds) | 58.37 | | 57.31 | 37.34 | |
| `doc-base` | same head, learned threshold | 57.55 | | | | |
| `doc-base` | global threshold chosen on dev | 63.15 | +5.60 | 61.49 | 45.0 | 63.04 |
| `doc-feat` | + pair features | 64.19 | +1.04 | 62.6 | 44.3 | 64.10 |
| `doc-tok2` | + 2 token layers | 64.75 | +1.60 | 63.1 | 46.5 | 64.49 |
| `doc-pair2` | + 2 pair layers | 63.99 | +0.84 | 62.5 | 45.6 | 63.90 |
| `doc-afl` | adaptive focal loss instead of ATL | 62.72 | −0.43 | 60.9 | 43.3 | 62.63 |
| `doc-full` | features + 2 token layers + 2 pair layers, learning rate 1e-4 | 65.99 | +2.84 | 64.3 | 48.2 | 65.85 |

Gains are against `doc-base` with the tuned threshold, except its own row (against the learned threshold).

- **The threshold is the largest single gain.** M3 was too cautious (precision 73, recall 48); a lower threshold trades precision for recall. "Cross-fitted" tunes the threshold on four folds of dev and scores the fifth: the gain holds (63.04).
- **Per-relation thresholds do not help:** 64.56 on the documents they were tuned on, 62.93 cross-fitted. Dropped.
- **The adaptive focal loss does not help,** on micro or macro F1. Dropped.
- **The three parts add up only partly:** +2.84 together, against +3.48 for the sum of the three.
- **Training stability:** token layers combined with another part diverge at learning rate 3e-4 (loss explodes in epoch 2). 1e-4 is stable. The diverged logs are in `results/logs/diverged/` and are not results.

## 3. Larger frozen encoders (dev, seed 0)

| Encoder | Base head | Full model |
|---|---|---|
| `bert-base-cased` | 63.15 | 65.99 |
| `roberta-large` | 61.04 | not run |
| `microsoft/deberta-v3-large` | 63.25 | 66.97 |

A larger frozen encoder brings nothing under the plain head, and +1.0 under the full model. Its main value is in an ensemble (section 5).

## 4. PromptORE as a typer (dev, seed 0)

`scripts/promptore_typing.py`: the document model (`doc-tok2`, F1 64.75) decides which pairs are related, and PromptORE only names the relation.

| Typing | PromptORE alone (v1) | On the pairs of `doc-tok2` | Typing F1 |
|---|---|---|---|
| Classifier (M2) | 49.22 | 60.91 | 88.4 |
| k-means clusters (M1) | 44.63 | 55.68 | 82.2 |

PromptORE lost about 11 points at detection alone. It stays below the document model, whose typing F1 is 92.2.

## 5. Combinations

`scripts/combine_runs.py` averages the fact scores of runs. The weight of two groups and the fact threshold are chosen on dev, so the dev column is slightly optimistic; test uses the dev choices.

| Run | Members | Dev F1 | Dev Ign F1 | **Test F1** | **Test Ign F1** | Test macro F1 |
|---|---|---|---|---|---|---|
| `atlop-frozen` | M3 of v1, 3 seeds | 58.37 | 57.31 | **57.70 ± 0.10** | **56.76** | 36.7 |
| `doc-full` | full model, bert-base, mean of 3 seeds | 66.18 ± 0.3 | 64.5 | **66.04 ± 0.1** | **64.5** | 48.5 |
| `doc-deberta-full` | full model, deberta-v3-large, seed 0 | 67.43 | 66.0 | **67.20** | **65.8** | 47.3 |
| `doc-full-ens` | the 3 seeds of `doc-full` | 67.68 | 66.1 | **67.4** | **65.9** | 50.3 |
| `doc-ens` | `doc-full-ens` + `doc-deberta-full`, weights 0.5 / 0.5 | 69.94 | 68.4 | **70.00** | **68.6** | 51.7 |
| `doc-ens+m2` | `doc-ens` + M2, weights 0.8 / 0.2 | 70.48 | 68.9 | **70.39** | **68.9** | 52.4 |

- Against M3 of v1 on test: **+8.3** for one full model, **+12.7** for the final ensemble.
- A second encoder in the ensemble brings more (+2.6 on test) than two more seeds of the same encoder (+1.4).
- M2 adds +0.4 on test: PromptORE and the document model are complementary, but only slightly once the document model is strong.
- Dev and test agree within 0.3 point on every row, so the choices made on dev did not overfit it.

The test rows are still `provisional`: they were computed once, with every choice made on dev, but the status is the team's decision (see [collaboration.md](collaboration.md)).

## 6. Partial fine-tuning of the encoder

`--tune-layers N` trains the last N layers of `bert-base-cased` with the relation model. The states that enter these layers are cached once (`doc_encoder.encode_lower`), so the lower layers are never run again and one epoch takes about 2 minutes (N = 2) to 3 minutes (N = 4). The model takes the output of the last layer and its attention (`doc_model.TunedEncoder`); the random token layers are no longer needed. Pair features and 2 pair layers are kept. Learning rate 3e-5 for the encoder layers, 1e-4 for the rest.

| Run | Members | Dev F1 | Dev Ign F1 | **Test F1** | **Test Ign F1** | Test macro F1 |
|---|---|---|---|---|---|---|
| `doc-full` | frozen encoder, one model (section 5) | 66.18 | 64.5 | **66.04** | **64.5** | 48.5 |
| `doc-tune2` | last 2 layers trained, 12 epochs, seed 0 | 71.04 | 69.5 | **70.88** | **69.4** | 55.9 |
| `doc-tune4` | last 4 layers trained, 15 epochs, seed 0 | 73.60 | 72.3 | **73.55** | **72.3** | 59.4 |
| `doc-tune4-s1` | last 4 layers trained, 12 epochs, seed 1 | 73.68 | 72.2 | **73.39** | **71.9** | 60.5 |
| `doc-tune-ens` | the two `doc-tune4` models (weight 0.8) + `doc-tune2` (0.2) | 74.34 | 72.9 | **74.31** | **73.0** | 60.5 |
| `doc-tune-ens+old` | `doc-tune-ens` (0.6) + `doc-ens+m2` (0.4) | 75.05 | 73.8 | **74.99** | **73.8** | 60.5 |

- **One model with 4 trained layers reaches test F1 73.5**, +7.5 over the frozen model and +3.2 over the ensemble of five frozen models (70.4). It trains in 50 minutes.
- **Each trained layer helps:** +4.8 for 2 layers, +2.7 more for 4. More layers were not tried.
- **Rare relations gain the most:** test macro F1 goes from 48.5 to 59.4.
- The two seeds of `doc-tune4` differ by 0.1 on dev and 0.2 on test, although one has 3 epochs less.
- The frozen ensemble still adds +0.7 on top of the fine-tuned models: the encoders differ (deberta, PromptORE).
- Dev F1 was flat over the last 3 epochs of `doc-tune4` (73.6): longer training is not the next gain.
- For reference, ATLOP fully fine-tuned on a GPU has 77.6 test F1 on Re-DocRED (BERT-base, published).

## 7. Limits

- Section 2 to 4 use one seed. The seeds of `doc-full` differ by about 0.3 point on dev, and the same seed retrained on MPS moved by 0.2 (bert-base) and 0.5 (deberta): gains below about 0.5 point are not established.
- The ensembles have one run each, so no seed interval.
- Partial fine-tuning (section 6) has one or two seeds per setting, and was only tried on `bert-base-cased` with 2 and 4 layers. `TunedEncoder` only supports BERT-like layers.
- Dev F1 was still rising slowly at epoch 20 for several frozen runs.

## 8. Reproduce

```bash
# one variant (options: --pair-features --token-layers N --pair-layers N --loss afl --encoder NAME)
python scripts/train_doc_model.py --run doc-full --pair-features --token-layers 2 --pair-layers 2 \
    --learning-rate 1e-4 --epochs 20 --seed 0 --test
python scripts/evaluate.py results/doc-full
python scripts/tune_thresholds.py results/doc-full
python scripts/promptore_typing.py results/doc-tok2
python scripts/combine_runs.py --run doc-ens --group results/doc-full:0 results/doc-full:1 results/doc-full:2 \
    --group results/doc-deberta-full:0
python scripts/combine_runs.py --run doc-ens+m2 --group results/doc-ens:0 --group results/promptore-classifier:0
# partial fine-tuning (section 6)
python scripts/train_doc_model.py --run doc-tune4 --tune-layers 4 --pair-features --pair-layers 2 \
    --learning-rate 1e-4 --epochs 15 --seed 0 --test
python scripts/combine_runs.py --run doc-tune-ens --group results/doc-tune4:0 results/doc-tune4-s1:1 \
    --group results/doc-tune2:0
python scripts/combine_runs.py --run doc-tune-ens+old --group results/doc-tune-ens:0 --group results/doc-ens+m2:0
```

`microsoft/deberta-v3-large` needs `sentencepiece` (in `requirements.txt`).
