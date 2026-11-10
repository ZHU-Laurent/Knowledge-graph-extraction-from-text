# Relation extraction with known entities: project plan

**Status:** current plan of the team, since 2026-11-09. The first working version is on the branch `known-entities-v1`, described in `docs/known_entities_v1.md` of that branch.

## 1. Scope

**Task:** document-level relation extraction with **known entities**. For each document, the entities are given with their mentions, types, coreference and Wikidata links. The system must say which relations hold between each ordered pair of entities, or that no relation holds. The predicted facts of all documents are then assembled into one knowledge graph.

**Not in scope:** entity detection, coreference resolution and entity linking. They come from the dataset annotations.

**Research questions:**

1. How well do relations discovered without labels, using [PromptORE](https://arxiv.org/abs/2304.01209), compare with supervised methods?
2. Where does each method fail: in **detection**, deciding that a pair has a relation, or in **typing**, choosing which relation?
3. How many labels does each method need?
4. How good is the graph built from the predicted facts, and can its confidence scores be trusted?

**Hypothesis:** detection is the hard part, not typing. PromptORE was designed for FewRel, where every instance has a relation, while 93% of entity pairs here have none. The first results (section 8) support this hypothesis.

## 2. Task definition

| Element | Definition |
|---|---|
| Input | A document, its sentences and its known entities: mentions, types, coreference and Wikidata links |
| Candidate | An ordered pair (head, tail) of different entities from the same document |
| Output | A set of relations for each candidate, from the 96 relations in `rel_info.json`, each with a score. An empty set means "no relation" |
| Fact | A (document, head, relation, tail) tuple |
| Graph | Facts of all documents merged into one graph, with a confidence per edge |

A pair can have several relations, and direction matters: (A, B) and (B, A) are different candidates.

**Detection and typing:** for analysis, and for some methods, the task is split in two:

- **Detection:** does the pair have at least one relation?
- **Typing:** given that it has one, which relations?

## 3. Data

We use **Linked-Re-DocRED** from the [Linked-DocRED repository](https://github.com/alteca/Linked-DocRED): DocRED, built from Wikipedia introductions, with the Re-DocRED relation fixes and links to Wikidata. See [personal_researches.md](personal_researches.md) for why we chose it.

| Split | Documents | Relation facts | Ordered entity pairs | Pairs with a relation | Use |
|---|---|---|---|---|---|
| Train | 3,053 | 85,932 | 1,193,092 | 67,808 (5.7%) | Training |
| Dev | 500 | 17,284 | 193,232 | 13,362 (6.9%) | Every choice of method or setting |
| Test | 500 | 17,448 | 198,670 | 13,627 (6.9%) | Final scores only |

The shared benchmark also fixes two small subsets for quick runs (seed 0): `train_debug`, 300 train documents, and `dev_debug`, 100 dev documents. See [collaboration.md](collaboration.md).

**Properties that shape the plan,** measured on test:

- **Most pairs have no relation** (about 93%), so a method must reject pairs as well as label them.
- **About half the related pairs cross sentences:** only 47.6% of related pairs share a sentence.
- **Several relations per pair:** 26.6% of related pairs have more than one label.
- **Direction:** 40.9% of related pairs are also related in reverse.
- **Rare relations:** the 10 largest relations hold 62.1% of facts, and 33 relation types have fewer than 50 instances.
- **Wikidata:** the relations are Wikidata properties, such as `P17` for "country". About 90% of entities other than dates and numbers have a Wikidata item, which merges them across documents in the graph. Dates and numbers have none.

We also use **FewRel** NYT and PubMed, the PromptORE paper's data, to check that our PromptORE implementation reproduces the paper.

## 4. Methods compared

| ID | Method | Representation of a pair | Labels used | Role |
|---|---|---|---|---|
| M0 | **Type-pair baseline** (`scripts/baseline_typepair.py`) | Entity types and whether the entities share a sentence, no text | Train | Lower bound: any method that does not beat it clearly is not reading the text |
| M1 | **PromptORE + clusters** | `[CLS] <context> <head> [MASK] <tail>. [SEP]` through a frozen `bert-base-uncased`, the `[MASK]` vector. One pass per pair | Train, for the detector and to name the clusters. None for the clusters | Closest to the paper |
| M2 | **PromptORE + classifier** | Same `[MASK]` vectors | Train | Same representation as M1, typed with labels directly. The gap with M1 is the cost of typing through clusters |
| M3 | **ATLOP-style classifier** | One pass of `bert-base-cased` per document, with entity and pair context vectors | Train | Supervised document-level reference, close to the state of the art |
| M4 | **LLM** (optional) | The document and the pair | None or a few examples | Generalist reference, on a dev sample |

### M1 and M2: PromptORE

- **Context:** the sentences of the two closest mentions of the pair (`mention_sentences`). It was as good as the shared sentence on dev, and also exists for pairs in different sentences.
- **Detector:** a logistic regression "has a relation", shared by M1 and M2. PromptORE assumes every instance has a relation, so pairs must be filtered first.
- **M1 typing:** k-means on the related train pairs, without their labels. Each cluster scores each relation by the share of its train pairs that have it, and a pair takes the scores of its nearest cluster. k is chosen on dev.
- **M2 typing:** one logistic regression per relation, on related pairs.
- **Score of a fact:** P(related) × score of the relation. The fact threshold is chosen on dev.
- **Training sample:** all 67,808 related train pairs and three times as many unrelated pairs (seed 0), with weights that restore the true share of unrelated pairs.
- **Label-free variant (optional):** name each cluster without labels, by comparing the words BERT predicts at `[MASK]` with the names and Wikidata aliases of the relations. It is the only fully unsupervised setting.

### M3: ATLOP-style classifier

- Each mention is marked, and each document goes once through `bert-base-cased`, frozen in v1.
- An entity vector pools its mentions, and a pair context vector pools the tokens that both entities attend to.
- A bilinear classifier with an **adaptive threshold** class handles "no relation" and pairs with several relations, without a threshold tuned on dev.
- **Next versions:** fine-tune the encoder, which is the most likely large gain, then try [DREEAM](https://github.com/YoumiMa/dreeam), which also predicts evidence sentences. Both need a GPU for long runs.

### M4: LLM (optional)

Published zero-shot results are weak, about 18 F1 on the Re-DocRED test set (section 6). We therefore test it where it is most useful: typing pairs accepted by a detector, or naming PromptORE clusters, on a fixed sample of dev documents.

## 5. Graph construction

- **Nodes:** an entity with a Wikidata item becomes the node of that item, shared across documents. Dates and numbers merge on their normalized value. Other entities stay local to their document.
- **Edges:** one per distinct (head node, relation, tail node). An edge keeps its supporting documents, and its confidence is the noisy-or of their scores: `1 − Π(1 − score)`.
- **Outputs:** the graph as JSON and GraphML, for networkx or Gephi.
- **Next steps:** parse dates, so "4 June 1985" and "1985-06-04" merge, and add consistency rules taken from Wikidata's property constraints (symmetric, inverse and single-value relations).

## 6. Evaluation

### Shared evaluator

**Decision of 2026-11-09:** every experiment is scored with the shared evaluator: `scripts/evaluate.py`, or its notebook version `notebooks/redocred_eval_cell.py`. See [evaluation.md](evaluation.md) and [collaboration.md](collaboration.md).

- Every method writes its predictions in the shared format and is scored on the fixed subsets.
- Results go to `report/leaderboard.csv`, so all methods are compared in one table.
- A metric of this section that the evaluator does not compute yet is added to the shared evaluator, not to separate code, and its notebook version is kept in sync.

### Metrics

| Metric | Definition | In the shared evaluator |
|---|---|---|
| **F1** | Micro precision, recall and F1 over predicted facts | Yes |
| **Ign F1** | F1 that ignores facts whose names already form a fact in train, the standard companion to F1 in DocRED papers | **To add** |
| **Macro F1** | Mean F1 over relation types, and by frequent, medium and rare relations | Yes |
| **Intra / inter** | Pairs that share a sentence, and pairs that do not | Recall only, F1 to add |
| **Detection F1** | F1 of the decision "this pair has a relation" | Yes |
| **Typing F1** | F1 on pairs that are related and detected | **To add** |
| **Evidence F1** | F1 on facts whose annotation lists evidence sentences | Yes |
| **B³, V-measure, ARI** | Clustering metrics of the PromptORE paper | Yes |
| **Graph metrics** | Graph precision, recall and F1 over distinct triples, precision of the N most confident edges, precision by number of supporting documents | **To add** |

### Rules

1. **Dev decides, test reports.** Every choice uses dev. Each method runs on test once, at the end, with its frozen settings.
2. **Seeds:** report the mean and standard deviation over 3 seeds for any method with randomness.
3. **Significance:** compare the main methods on test with a paired bootstrap over documents.
4. **Same candidates for everyone:** all ordered pairs of known entities. A method that skips pairs predicts "no relation" for them.

### Published reference points

Re-DocRED test scores with known entities, read on 2026-11-08 from the papers' arXiv versions. They use RoBERTa-large, according to the DREEAM paper, without extra distantly labeled data. Our models use BERT-base, so expect lower scores.

| Model | Ign F1 | F1 | Source |
|---|---|---|---|
| JEREX | 71.45 | 72.57 | [Re-DocRED](https://arxiv.org/abs/2205.12696), Table 6 |
| ATLOP | 76.82 | 77.56 | Re-DocRED, Table 6 |
| KD-DocRE | 77.60 | 78.28 | Re-DocRED, Table 6 |
| DREEAM | 77.34 | 77.94 | [DREEAM](https://arxiv.org/abs/2302.08675), Table 6 |
| LMRC (LLaMA2-13B) | 74.08 | 74.63 | [LMRC](https://arxiv.org/abs/2408.13889), Table 3 |
| Mistral-Large 123B, zero-shot | 18.50 | 18.61 | [GLiDRE](https://arxiv.org/abs/2508.00757), Table 4 |

Check the DREEAM numbers and the encoder of Table 6 in the PDFs before the report.

## 7. Experiments

| ID | Experiment | Question |
|---|---|---|
| A | Compare M0 to M4 on test, after every choice is made on dev | Best method, detection versus typing |
| B | Repeat M1, M2 and M3 with labels from 1%, 5%, 10%, 25% and 100% of train documents, 3 samples per size | How many labels each method needs |
| C | PromptORE ablations on dev: context, prompt, k, encoder | Which PromptORE settings matter |
| D | Analyses on dev: errors by relation frequency and by sentence distance, inverse relations, false positives checked in Wikidata and by hand, entity names replaced by their type, calibration of the scores | Why methods fail, and whether the scores can be trusted |
| E | Build and analyze the graph of each method | Graph quality and what repetition across documents brings |

## 8. Status

**Done:**

- PromptORE reproduced on FewRel PubMed, within 1.3 points of the paper.
- Shared benchmark and evaluator, with the type-pair baseline (dev F1 18.8).
- On `known-entities-v1`, a first version of M1, M2 and M3 and of the graph. Results on test, one seed:

| Method | F1 | Ign F1 | Detection F1 | Typing F1 | Graph F1 |
|---|---|---|---|---|---|
| M3, ATLOP-style, frozen encoder | 57.76 | 56.75 | 60.11 | 92.25 | 59.18 |
| M2, PromptORE + classifier | 50.45 | 48.09 | 53.25 | 89.02 | 50.72 |
| M1, PromptORE + clusters (k = 400) | 46.37 | 44.10 | 51.21 | 84.14 | 46.48 |

Typing F1 is between 84 and 92 while detection F1 is between 51 and 60: detection is the hard part for every method.

### Next steps

| # | Step | Depends on |
|---|---|---|
| 1 | Merge `known-entities-v1` into `main` through a pull request, and score its methods with the shared evaluator | |
| 2 | Add Ign F1, typing F1, intra / inter F1 and the graph metrics to the shared evaluator | |
| 3 | Finish the FewRel reproduction on NYT | |
| 4 | Fine-tune the encoder of M3, on a GPU | 1 |
| 5 | Experiment B, label efficiency | 1, 2 |
| 6 | Experiments C and D | 1, 2 |
| 7 | Run 3 seeds per method, then test once | 4 to 6 |
| 8 | Optional: label-free cluster naming, DREEAM, LLM | 1 |
| 9 | Report and slides | All above |

## 9. Risks

| Risk | Mitigation |
|---|---|
| Low recall of every method | Detection is the bottleneck: try a lower fact threshold, chosen on dev, and a fine-tuned encoder |
| Rare relations | Report results by frequency |
| No GPU for fine-tuning | Cloud GPU, otherwise keep the frozen encoder and cite published scores |
| PromptORE is slow: one BERT pass per pair | Cache the embeddings in float16 and reuse the same train sample |
| Missing annotations counted as errors | Check a sample of false positives in Wikidata and by hand |

## 10. Licenses

The PromptORE and Linked-DocRED code, and the Linked-DocRED data, are under GPL-3.0. We keep the original code in `external/` and the data in `data/raw/`, both outside Git, and write our own implementations.

## 11. To confirm with the teacher

- Is Linked-Re-DocRED acceptable in place of Linked-DocRED?
- Is a GPU available to fine-tune the encoder?
- Should the LLM comparison be included?
- What are the expected deliverables and deadlines?
