# Knowledge graph project: plan

**Scope, agreed with the teacher:** use a dataset labeled with entities and relations, find relations with PromptORE or another method, and assign each discovered cluster to a relation from a fixed list that includes "no relation".

**Research question:** how well can [PromptORE](https://arxiv.org/abs/2304.01209) clusters, mapped to a fixed list of relations, recover labeled relations and reject pairs without a relation, compared with a classifier trained on the same embeddings?

**Out of scope:** entity detection, entity normalization and graph construction as a main goal. Entities come from the dataset, so predicted triples can already form a graph if we want one for the demo.

## Dataset

We use **Linked-Re-DocRED** from the [Linked-DocRED repository](https://github.com/alteca/Linked-DocRED). See [personal_researches.md](personal_researches.md) for the measurements.

- **Relation list:** the 96 relations in `rel_info.json`, plus "no relation".
- **Why this variant:** the Linked-DocRED test set has no relation labels, and DocRED's incomplete annotations would make many real relations look like "no relation". Linked-Re-DocRED labels its test set and doubles the number of relations.

## Pipeline

| # | Stage | First version |
|---|---|---|
| 1 | Load the data | Documents, gold entities and gold relations. Keep the official splits |
| 2 | Build candidates | Every ordered entity pair in a document, removing entity type pairs never seen with a relation in train |
| 3 | Embed | PromptORE's `[MASK]` embedding from an untrained BERT, with the document as context |
| 4 | Cluster | k-means on train embeddings, fitted without labels. Dev decides k, testing values above the number of relations |
| 5 | Map clusters to relations | Use train labels to link each cluster to relations whose precision in that cluster passes a threshold. Other clusters map to "no relation". Dev sets the threshold |
| 6 | Predict | Assign each dev or test pair to its nearest cluster and output that cluster's relations |
| 7 | Evaluate and compare | See below |

## Design decisions

Measured on the Linked-DocRED and Linked-Re-DocRED files.

| Issue | Evidence (Linked-Re-DocRED test) | Decision |
|---|---|---|
| PromptORE works on sentences, but these relations are document-level | Only 47.6% of related pairs share a sentence | Use the document as prompt context. Compare with the sentence-level version, which limits recall |
| Pairs without a relation dominate | 6.9% of ordered pairs have a relation | Majority vote would map almost every cluster to "no relation", so we use a precision threshold instead |
| Several relations per pair | 26.6% of related pairs have more than one label | A cluster can map to several relations |
| Relation direction | 40.9% of related pairs are also related in reverse | Candidates are ordered pairs, and scoring takes direction into account |
| Rare relations | The 10 largest relations hold 62.1% of instances, and 33 have fewer than 50 | Report micro and macro F1, plus results by relation frequency |
| Prompt length | Documents average 197 words, and BERT accepts 512 tokens | Truncate only the document, never the prompt, and report how many are truncated |
| Number of pairs | Train has about 1.2 million ordered pairs | Remove impossible type pairs, then fit k-means on a sample. We compare a random sample, which uses no labels, with one balanced using train labels |
| Multiple mentions | An entity can appear under several names | Use its longest mention in the prompt in the first version |
| Remaining annotation gaps | Re-DocRED is more complete, but may still miss relations | Review a sample of false positives by hand |
| Code license | The PromptORE and Linked-DocRED code is under GPL-3.0 | Keep the original code in `external/`, and write our own version from the papers. Check the license if any of it enters this repository |

The clusters use no labels, but the mapping does. This setup is therefore not fully unsupervised, and the report must say so.

## Evaluation

**Main score:** precision, recall and F1 on predicted (document, head, relation, tail) facts, excluding "no relation", with micro and macro averages. We cross-check our score against the repository's `relation_f1.py`.

**Also reported:**

- B³, V-measure and ARI on related pairs, to compare with the PromptORE paper.
- Share of pairs without a relation in each cluster.
- Errors by relation frequency, shared sentence or not, and number of labels.

**Comparisons:**

| Method | Purpose |
|---|---|
| Sentence-level PromptORE | Shows what document context adds |
| Linear classifier on the same embeddings | Shows the cost of going through clusters rather than training directly on labels |
| ATLOP from the Linked-DocRED repository (optional) | Gives a supervised state-of-the-art reference |

**Avoiding leakage:** make every decision on dev, and score test only once at the end.

## Work plan

1. **Reproduce PromptORE on FewRel.** Run the original code in its own environment, starting with FewRel PubMed. With BERT and k = 80, the paper reports a B³ F1 of 48.8, a V-measure F1 of 71.8 and an ARI of 43.4 on FewRel. The paper's prompt ends with a period and the main script's does not, so we test both.
2. **Load Linked-Re-DocRED** and build the candidates. Count pairs after type filtering, and check that no real relation is lost on train.
3. **Cluster related pairs only.** Compare sentence and document context with the clustering metrics.
4. **Add pairs without a relation** and measure how they affect the clusters.
5. **Map clusters to relations** and tune k and the threshold on dev.
6. **Train the linear classifier** on the same embeddings.
7. **Analyze errors** on dev, then run the final test evaluation once.
8. **Write the report** and prepare a short demo.

## Main risks

- **Mixed clusters:** pairs without a relation can contaminate real clusters. A high threshold then loses recall, and a low one loses precision.
- **Rare relations:** many are unlikely to get their own cluster, so macro F1 may stay low.
- **Compute:** embedding hundreds of thousands of pairs may need a GPU. Start with a subset of documents.
- **Document context:** the paper only tests sentences, so longer context may not help.

## Optional extensions

- Map clusters without labels, by comparing them with relation names and descriptions in `rel_info.json`.
- Use an LLM to match clusters to the relation list.
- Test the alternative prompts from the paper.
- Show predicted triples as a graph.

## To confirm with the teacher

- Is Linked-Re-DocRED acceptable in place of Linked-DocRED?
- Should we keep all 96 relations, or use a shorter list?
- What are the expected deliverables and deadlines?
