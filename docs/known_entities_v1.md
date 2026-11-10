# Known entities v1: relation detection and graph construction

**Origin:** branch `known-entities-v1`. **Status:** first working version, compared with two PromptORE methods. Test: relation F1 57.76, graph F1 59.18 (section 6). Comparison in section 7.

This is the first version of the methods of the project plan ([project_plan.md](project_plan.md)). It assumes that the entities of each document are known, and builds only two things: the detection of relations between these entities, and a knowledge graph of the whole corpus. Its main method is an ATLOP-style classifier. Section 7 compares it with two methods built on PromptORE, the baseline paper of the project.

This document explains what happens at each step: dataset, processing, training, evaluation and graph construction.

```text
1. Dataset      Linked-Re-DocRED documents, with their known entities
2. Processing   each document encoded once by a frozen BERT, one vector per entity and per pair
3. Training     a classifier that gives each ordered pair of entities its set of relations
4. Evaluation   relation metrics per document, then graph metrics
5. Graph        facts of all documents merged into one graph, with a confidence per edge
```

## 1. Dataset

### Source

We use **Linked-Re-DocRED**, from the [Linked-DocRED repository](https://github.com/alteca/Linked-DocRED). Each document is the introduction of a Wikipedia article. The annotations come from DocRED, with the relation fixes of Re-DocRED and links to Wikidata.

| Split | Documents | Relation facts | Ordered entity pairs | Pairs with a relation |
|---|---|---|---|---|
| Train | 3,053 | 85,932 | 1,193,092 | 5.7% |
| Dev | 500 | 17,284 | 193,232 | 6.9% |
| Test | 500 | 17,448 | 198,670 | 6.9% |

The relations are 96 Wikidata properties, listed in `rel_info.json`, for example `P17` "country" or `P19` "place of birth".

### What a document contains

```text
title       "Willi Schneider (skeleton racer)"
sents       list of sentences, each a list of tokens
entities    list of entities, each with:
              id, type (PER, ORG, LOC, TIME, NUM, MISC)
              mentions: name, sentence index, token span
              entity_linking: Wikidata item, for example Q321314
relations   list of facts: head id, tail id, relation code, evidence sentences
```

An entity groups all its mentions. In the example, "Wilfried " Willi " Schneider" and three mentions "Schneider" are one `PER` entity.

### What "known entities" means here

The system receives `title`, `sents` and `entities` as annotated: mentions, types, coreference and Wikidata links are all given. It must predict `relations`. The gold `relations` are only used for training on train, for choosing the best epoch on dev, and for the final scores.

### What makes the task hard

- **Most pairs have no relation:** about 93% of ordered pairs. The model must reject pairs, not only label them.
- **Several relations per pair:** 26.6% of related test pairs have more than one relation.
- **Direction matters:** (A, B) and (B, A) are two different pairs.
- **Relations across sentences:** about half of the related pairs never appear in the same sentence.

## 2. Processing

The goal of processing is to turn each document into vectors: one per entity, and one context vector per ordered pair. This is done once, and the result is reused at every training epoch.

Code: `src/kg_extraction/relations/doc_encoder.py`, then `corpus_inputs` in `pair_classifier.py`.

### 2.1 Mark the mentions

Every mention is wrapped in `*` markers before the text is split into word pieces. The start of the example document becomes:

```text
* W ##il ##fried " Will ##i " Schneider * ( born 13 March 1963 in * Mediaș * ...
```

The position of each opening `*` is recorded, grouped by entity. The `*` token will stand for its mention: its vector, once encoded, has seen the whole mention and its context. This idea comes from ATLOP (Zhou et al., 2021).

### 2.2 Encode each document once

The marked document goes through `bert-base-cased`, whose weights are **frozen**: they are never trained.

- **One pass per document,** not one per pair. The whole dataset needs about 4,000 passes, instead of 1.19 million if each pair had its own input.
- **Token vectors:** the mean of the last 4 layers of BERT. The last layer alone is specialized for predicting masked words.
- **Attention:** for each mention marker, the attention it pays to every token of the document, taken from the last layer and averaged over its 12 heads.
- **Long documents:** BERT reads at most 512 tokens. A longer document is cut into windows of 512 tokens that overlap by half, and the vectors and attention of overlapping tokens are averaged. Only 14 of the 500 dev documents need this.

The result is cached in `data/processed/features/<split>.pt`. Encoding took about 4 minutes for train and 30 seconds for dev on the Mac GPU.

**Why frozen:** fine-tuning BERT on whole documents needs long GPU runs, and the Mac GPU was shared with other runs. A frozen encoder gives a working v1 quickly. It is also weaker, so fine-tuning is the first improvement to try.

### 2.3 One vector per entity

An entity's vector is the **logsumexp** of its mention vectors: a smooth maximum. The most informative mention dominates, and the other mentions still count.

### 2.4 One context vector per pair

For a pair (head, tail):

1. Take the attention of the head entity over the document (mean of its mentions' attention), and the same for the tail.
2. Multiply them token by token, and normalize: tokens get a high weight when **both** entities attend to them.
3. The context vector is the weighted mean of the token vectors.

This finds, for each pair, the words that link the two entities, even when they are in different sentences, without choosing sentences by hand.

### 2.5 Output of processing = input of training

Processing has two parts:

| Part | Runs | Input | Output |
|---|---|---|---|
| Encoding (2.1 and 2.2) | Once, then cached on disk | The tokens of a document and its mention positions | For each document: one vector per token (tokens × 768) and one attention row per mention (mentions × tokens) |
| Pooling (2.3 and 2.4) | At the start of each training or prediction run, in memory | The cached encoding | For each ordered pair: 3 vectors of size 768 |

A document with 20 entities has 20 × 19 = 380 ordered pairs, so it becomes a table of shape (380, 3, 768): for each pair, the head entity vector, the tail entity vector and the pair context vector.

**Training receives only these tables and the labels.** It never sees the text or BERT:

```text
inputs   (1,193,092 train pairs, 3, 768)   from processing, stored in half precision (about 1.8 GB)
labels   (1,193,092 train pairs, 97)       from the gold relations, see 3.1
```

All train pairs are stacked into one table, regardless of their document, and the classifier learns to map the 3 vectors of a pair to its 97 labels.

## 3. Training

Code: `PairClassifier` and `adaptive_threshold_loss` in `pair_classifier.py`, and `scripts/train_relation_model.py`.

### 3.1 Labels

Each pair gets a vector of 97 zeros and ones: one column per relation, plus column 0, set to 1 when the pair has no relation. A pair with "country" and "located in" has two ones.

### 3.2 The classifier

```text
head vector  = tanh(W_h [head entity ; pair context])
tail vector  = tanh(W_t [tail entity ; pair context])
logits (97)  = grouped bilinear(head vector, tail vector)
```

- **Separate weights for head and tail,** so (A, B) and (B, A) get different scores.
- **Grouped bilinear layer:** the two vectors are cut into 12 blocks of 64, and every product between two values of the same block is fed to a linear layer. It captures how head and tail combine, with far fewer weights than a full bilinear layer.

### 3.3 The adaptive threshold

Column 0 is not a "no relation" class like the others. It is a **threshold learned for each pair:** a relation is predicted when its logit is above the logit of column 0.

The loss, from ATLOP, has two parts:

- each gold relation of the pair must score above the threshold;
- the threshold must score above every relation that the pair does not have.

This handles the 93% of pairs without a relation (no relation passes the threshold) and pairs with several relations (several pass it). No global threshold needs to be tuned.

Each predicted fact gets the score `sigmoid(relation logit − threshold logit)`, between 0.5 and 1 for predicted facts. The graph uses it as the fact's confidence.

### 3.4 The training loop

| Setting | Value |
|---|---|
| Training data | All 1.19 million train pairs, shuffled across documents |
| Batch | 1,024 pairs |
| Optimizer | AdamW, learning rate 3e-4, one-cycle schedule, gradient clipping at 1.0 |
| Regularization | Dropout 0.1 on the input vectors |
| Length | Up to 30 epochs. Stops after 5 epochs without a better dev F1 |
| Model kept | The epoch with the best dev F1, saved to `data/processed/models/relation_model.pt` |

These are usual settings for a small head, not tuned.

**Note on speed:** a first version computed the entity and context vectors inside the training loop, document by document. It took 1.1 s per batch of 4 documents on the Mac GPU, about 14 minutes per epoch. Since the encoder is frozen, these vectors never change, so they are now computed once before training.

## 4. Evaluation

Code: `src/kg_extraction/evaluation/relations.py` (from `main`) and `src/kg_extraction/evaluation/graph.py`.

**Rule:** dev chooses the epoch, test is scored once at the end.

### 4.1 Relations, document by document

A fact is (document, head entity, relation, tail entity). Predicted facts are compared with gold facts, using the standard DocRED metrics:

| Metric | What it measures |
|---|---|
| Precision, recall, F1 | Over all facts of the split |
| Ign F1 | F1 without the facts whose names already appear as a fact in train, so memorized facts do not count |
| Intra / inter F1 | F1 on pairs that share a sentence, and on pairs that do not |
| Detection F1 | Whether the model finds that a pair has a relation, whatever the relation |

### 4.2 Graph

The gold graph is built from the gold facts with the same rules as the predicted graph (section 5). Then:

| Metric | What it measures |
|---|---|
| Graph precision, recall, F1 | Over distinct (head node, relation, tail node) triples, after merging documents |
| Precision at N | Precision of the N edges with the highest confidence, for N = 100, 500, 1,000 and 5,000 |
| Precision by support | Precision of edges found in 1, 2, or 3 or more documents: does repetition make a fact more reliable? |

The report also describes the graph: number of nodes and edges, linked nodes, connected components, most connected nodes and most frequent relations.

## 5. Graph construction

Code: `src/kg_extraction/graph/build.py` and `scripts/build_graph.py`.

### 5.1 Nodes

Every entity is mapped to a node key:

| Entity | Node key | Example |
|---|---|---|
| Linked to a Wikidata item | `wd:<item>` | `wd:Q84` for London, in every document |
| `TIME` or `NUM` | `time:<text>` or `num:<text>`: longest mention, lower case, punctuation removed | `time:13 march 1963` |
| Other entities | `doc:<title>#<id>`, local to its document | `doc:Willi Schneider (skeleton racer)#5` |

- Entities with the same Wikidata item become **one node across documents.** This is what connects the documents into one graph.
- Dates and numbers have no Wikidata link in the dataset, so they merge when their text matches.
- Entities without a link stay in their document: merging them on the name alone could join different entities.

Each node keeps its most frequent name, its type, its Wikidata item and the documents where it appears.

### 5.2 Edges

1. Each predicted fact is rewritten with node keys: (head node, relation, tail node).
2. Facts whose head and tail fall on the same node are dropped.
3. Facts with the same triple are merged into **one edge.** If the same document gives the triple twice, its best score is kept.
4. The edge's confidence is the **noisy-or** of its document scores: `1 − Π(1 − score)`. Two documents at 0.5 give 0.75. A fact found in several documents becomes more certain.
5. The edge also keeps its best single score, its number of supporting documents, and the title and score of each document, so it can be traced back to the text.

### 5.3 Output

`scripts/build_graph.py --split dev` (or `test`) writes to `results/known_entities/`:

| File | Content |
|---|---|
| `<split>_predictions.json` | The documents in the Linked-DocRED format, with the predicted relations and their scores |
| `<split>_graph.json` | Nodes and edges, sorted by confidence |
| `<split>_graph.graphml` | The same graph for networkx or Gephi |
| `<split>_report.json` | Relation metrics, graph metrics and graph summary |

## 6. Results

One run, seed 0, on 2026-11-09. The best dev F1 was reached at epoch 19. Training stopped at epoch 24, after 5 epochs without a better dev F1. With the precomputed pair vectors, an epoch took about 1.5 minutes on the Mac GPU. Test was scored once, with the epoch 19 model.

### Relations, per document

| Split | Precision | Recall | F1 | Ign F1 | Intra F1 | Inter F1 | Detection F1 | Predicted / gold facts |
|---|---|---|---|---|---|---|---|---|
| Dev | 72.92 | 49.20 | 58.75 | 57.62 | 61.27 | 56.55 | 61.21 | 11,661 / 17,284 |
| Test | 73.73 | 47.48 | 57.76 | 56.75 | 60.82 | 55.02 | 60.11 | 11,235 / 17,448 |

- **Precision is much higher than recall.** The model misses about half the gold facts, and most of the loss is in detection: detection F1 is only 2 to 3 points above F1, so when a related pair is found, its relation is usually right.
- **Pairs across sentences** score only 4 to 6 points below pairs in the same sentence, so the pair context does reach other sentences.
- **Across epochs,** dev F1 moved between 54 and 59 from epoch 4 on, with precision and recall trading places from one epoch to the next.

### Graph

| Split | Nodes | Linked to Wikidata | Edges | Components | Largest component |
|---|---|---|---|---|---|
| Dev | 4,508 | 3,637 | 10,576 | 96 | 3,977 nodes |
| Test | 4,464 | 3,582 | 10,203 | 110 | 3,854 nodes |

| Split | Precision | Recall | F1 | P@100 | P@1,000 | P@5,000 | Precision, 1 document | 2 documents | 3 or more |
|---|---|---|---|---|---|---|---|---|---|
| Dev | 72.42 | 50.75 | 59.68 | 100.0 | 97.9 | 88.64 | 71.90 | 96.95 | 100.0 |
| Test | 73.65 | 49.46 | 59.18 | 100.0 | 98.1 | 90.14 | 73.13 | 95.65 | 98.44 |

- **Confidence ranks well:** the 1,000 most confident test edges are 98.1% correct, against 73.65% for all edges.
- **Facts found in several documents are almost always right,** but they are few: on test, 9,978 edges come from one document, 161 from two and 64 from three or more.
- **Most nodes form one large component,** connected through countries and nationalities: the most connected test nodes are "American" (1,208 edges), "Canada", "German", "India" and "British".
- **The most frequent relations** are "located in the administrative territorial entity" and "country", as in the annotations.

**Bug found and fixed:** in the dataset, dates and numbers have the Wikidata value `#ignored#`, and entities without an item have placeholders such as `#DocRED-246#`. The first graph took them for items, so all dates and numbers merged into one node. Only identifiers of the form `Q<number>` now count as links. The scores above are after the fix.

## 7. Comparison with PromptORE

The relation step of the pipeline was also run with two methods built on [PromptORE](https://arxiv.org/abs/2304.01209), the baseline paper of the project. Everything else is the same: known entities, graph rules and metrics. Code: `scripts/compare_methods.py`.

### The three methods

| Method | Representation of a pair | Detection | Typing | Labels used |
|---|---|---|---|---|
| **ATLOP** (sections 2 and 3) | Entity and context vectors from one pass of a frozen `bert-base-cased` per document | Learned threshold of each pair | Same classifier | Train, all pairs |
| **PromptORE + classifier** | `[CLS] <context> <head> [MASK] <tail>. [SEP]` through a frozen `bert-base-uncased`, the `[MASK]` vector. One pass per pair. Context: the sentences of the two closest mentions | Logistic regression "has a relation" | One logistic regression per relation, on related pairs | Train, sample of pairs |
| **PromptORE + clusters** | Same `[MASK]` vectors | Same detector | k-means on related train pairs, without labels. Each cluster scores each relation by the share of its train pairs that have it. A pair takes the scores of its nearest cluster | Train, for the detector and to name the clusters |

- **PromptORE + clusters** is the closest to the paper: relations come from unsupervised clusters. The paper assumes every instance has a relation, which is false here (93% of pairs have none), so a detector filters pairs first.
- **PromptORE + classifier** uses the same vectors, but types pairs with labels directly. The gap between the two PromptORE methods measures the cost of typing through clusters.
- Both PromptORE methods score a fact as `P(related) × score of the relation`. They differ only in the second factor.
- **PromptORE training sample:** all 67,808 related train pairs and three times as many unrelated pairs, drawn at random (seed 0), with weights that restore the true share of unrelated pairs. Pairs whose two entity types never carry a relation in train are skipped and count as "no relation". This sample and its embeddings are the same as for the probe of `main` (`scripts/train_relation_probe.py`).
- **Settings chosen on dev:** the fact threshold of both PromptORE methods (0.18 and 0.17), and the number of clusters, among 100, 200 and 400 (dev F1 41.95, 43.47 and 44.15, so k = 400). ATLOP has no threshold set on dev: it uses its learned threshold.

### Results on test

One run, seed 0. Test was scored once, with the settings chosen on dev.

| Method | Precision | Recall | F1 | Ign F1 | Intra F1 | Inter F1 | Detection F1 | Typing F1 | Graph F1 | Graph P@1,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| ATLOP | 73.73 | 47.48 | **57.76** | **56.75** | **60.82** | **55.02** | **60.11** | **92.25** | **59.18** | **98.1** |
| PromptORE + classifier | 49.18 | 51.78 | 50.45 | 48.09 | 55.24 | 46.81 | 53.25 | 89.02 | 50.72 | 94.9 |
| PromptORE + clusters | 46.77 | 45.97 | 46.37 | 44.10 | 50.14 | 43.47 | 51.21 | 84.14 | 46.48 | 94.7 |

On dev, F1 is 58.75, 49.24 and 44.15, in the same order.

**Typing F1** only keeps the pairs that are related in the gold annotations and detected by the method, so it measures the choice of relation alone. Missed pairs and false pairs are counted by detection F1.

### What the comparison shows

- **Detection is the hard part, for every method.** Typing F1 is between 84 and 92, while detection F1 is between 51 and 60. When a method finds a related pair, it usually gives it the right relation. This confirms the hypothesis of the main plan.
- **ATLOP wins mainly on detection:** 60.1 against 53.3 and 51.2. Its precision is much higher (73.7 against 49.2 and 46.8), for a similar recall.
- **Clusters cost about 5 points of typing F1** against the classifier on the same vectors (84.1 against 89.0), and 4 points of F1. Typing through 400 unsupervised clusters keeps most of the information.
- **Pairs across sentences** are where PromptORE loses most: inter F1 is 8 to 12 points below ATLOP, against 6 to 11 points for intra F1. The PromptORE context only holds the sentences of the two closest mentions, while ATLOP reads the whole document.
- **Graphs:** ATLOP gives the smallest and most precise graph (10,203 edges). The PromptORE graphs have about 16,600 and 15,500 edges, with graph precision of 48.6 and 46.0. All three rank their edges well: the 1,000 most confident edges are 95% to 98% correct.
- **Cost:** PromptORE needs one BERT pass per pair. Embedding the 198,000 test pairs took about 1.5 hours on the Mac GPU, against about 30 seconds to encode the 500 test documents for ATLOP.

### Limits of this comparison

- **The encoders differ:** `bert-base-cased` for ATLOP, `bert-base-uncased` for PromptORE as in the paper, and ATLOP uses the mean of the last 4 layers.
- **Thresholds differ:** the PromptORE methods take theirs from dev, ATLOP learns one. Setting ATLOP's on dev could only help it.
- **One seed:** no standard deviation, and no significance test yet.

## 8. Limits and next steps

- **Frozen encoder:** fine-tuning the encoder is the most likely large gain, mainly for recall.
- **Low recall:** the learned threshold is strict. Lowering the score needed to predict a fact, chosen on dev, would trade precision for recall.
- **Dates and numbers:** they merge on their text only, so "4 June 1985" and "1985-06-04" stay separate nodes. A date parser would merge them.
- **No consistency checks:** symmetric and inverse relations, and single-value relations such as a place of birth, are not used yet.
- **Settings not tuned:** layers, learning rate and block size are usual defaults.
- **Known entities:** a real system must find the entities itself, which this branch does not cover.

## 9. Files

| File | Role |
|---|---|
| `src/kg_extraction/relations/doc_encoder.py` | Mention markers, document encoding, cache |
| `src/kg_extraction/relations/pair_classifier.py` | Entity and pair vectors, classifier, loss, prediction |
| `src/kg_extraction/graph/build.py` | Node keys, edge merging, networkx export |
| `src/kg_extraction/evaluation/graph.py` | Graph metrics and summary |
| `scripts/train_relation_model.py` | Training |
| `scripts/build_graph.py` | Prediction, graph and report for dev or test |
| `src/kg_extraction/relations/encoder.py`, `cache.py`, `probe.py`, `prompts.py` | PromptORE embeddings and classifier, from `main` |
| `src/kg_extraction/relations/clustering.py` | k-means and the mapping of clusters to relations |
| `scripts/compare_methods.py` | The comparison of section 7 |
| `tests/test_known_entities.py` | Unit tests |
