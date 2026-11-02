# Knowledge graph project: plan

**Goal:** build a knowledge graph from raw text with a pipeline that needs no annotations from the target domain, using [PromptORE](https://arxiv.org/abs/2304.01209) as its relation step.

**Research question:** how far can an unsupervised, domain-independent pipeline built around PromptORE go from raw text to a usable knowledge graph, and how much quality does each stage lose?

## Why PromptORE needs a pipeline

PromptORE embeds each entity pair with a prompt and an untrained BERT, then clusters the embeddings to discover relations. On its own, it:

- expects the entities as input;
- assumes every pair expresses a relation;
- returns unnamed clusters, not named triples.

The upstream stages supply the entity pairs. The downstream stages turn the clusters into triples and a graph.

## Pipeline

| # | Stage | First version | Later improvement |
|---|---|---|---|
| 1 | Split into sentences | spaCy | |
| 2 | Detect entities | GLiNER, with entity types given as a per-corpus setting | |
| 3 | Build candidate pairs | Every pair in the same sentence, in order of appearance | Limit pairs by distance |
| 4 | Filter pairs without a relation | None: PromptORE receives every pair | Rules, then an NLI model or an LLM |
| 5 | Embed and cluster | PromptORE prompt with BERT, k-means, and k estimated with the elbow rule | Alternative prompts |
| 6 | Filter the results | Remove instances far from their cluster centre, using a fixed percentile | Remove clusters with low cohesion |
| 7 | Name the clusters | Give an LLM the sentences closest to each centre; it returns a name or "no clear relation". Merge clusters with the same name | Decide the direction of each instance |
| 8 | Build triples | (entity 1, cluster name, entity 2), with the source sentence and a confidence score. Drop pronouns and generic mentions | Add coreference resolution |
| 9 | Normalize entities and build the graph | Merge mentions with simple rules, then store with NetworkX | Embedding similarity, Wikidata linking |

## Design decisions

These decisions come from reviewing the pipeline against the paper, the PromptORE code and the SciERC data. The SciERC figures were measured on an unofficial copy and should be measured again on the official files.

| Issue | Evidence | Decision |
|---|---|---|
| Pairs without a relation | 38% of the 12,211 same-sentence pairs in SciERC have a relation, so 62% do not | Stage 4 starts empty and serves as our no-filter baseline. These pairs may not form their own clusters and can contaminate real ones, so we measure the share of them in each cluster |
| Unbalanced relations | `USED-FOR` is 52.4% of SciERC relations, while FewRel is perfectly balanced | k-means may split large relations, so stage 7 merges clusters with the same name. We report results before and after merging |
| Relation direction | Pairs follow order of appearance, but the text can state a relation in either direction | The first version is scored without direction. Direction is a later improvement |
| Entities that make poor nodes | SciERC labels 1,338 of 8,094 entities as `Generic`, and about 183 look like pronouns | Stage 8 drops them, and we report how many triples this removes. Coreference resolution could recover them later |
| Long sentences | The longest SciERC sentence has 101 tokens. The PromptORE code truncates the whole prompt, and fails if `[MASK]` is cut off | Our version truncates only the sentence and always keeps the prompt |
| Sentence-level extraction | No SciERC relation crosses a sentence boundary | Same-sentence pairs lose no SciERC relation. Other corpora may differ |
| Entity types per corpus | GLiNER needs a list of entity types | This is the only per-corpus setting. It needs no annotations or training |
| Tuning without labels | Using labels to set thresholds would break the unsupervised goal | Thresholds stay fixed. Labels only measure how sensitive the results are |
| Code license | The PromptORE code is under GPL-3.0 | We run the original outside this repository to reproduce the paper, and write our own version from the paper's description |

## Evaluation

| Stage | Measure |
|---|---|
| Entities | Precision, recall and F1 |
| Pair filter | Share of real relations kept and of other pairs removed |
| Clustering | B³, V-measure and ARI, with gold pairs, then all gold-entity pairs, then predicted entities |
| Cluster names | Manual review of a sample |
| Triples | Precision, recall and F1 after mapping each cluster to its majority relation |

**Avoiding leakage on SciERC:**

- Fit the clustering on all abstracts, since this step uses no labels.
- Map clusters to relations using the train and dev labels only.
- Make decisions on dev, and report final triple scores on test only once.

The cluster mapping uses labels, so it is for evaluation only and gives an upper bound on what the unsupervised pipeline achieves. Clustering scores and supervised F1 scores from other papers are not directly comparable.

## Work plan

Each milestone adds one stage, so we can measure its cost and always have a result to present.

1. **Reproduce PromptORE on FewRel.** Run the original code in its own environment, starting with FewRel PubMed and NYT as quick tests. With BERT and k = 80, the paper reports a B³ F1 of 48.8, a V-measure F1 of 71.8 and an ARI of 43.4. Record any gap. The paper's prompt ends with a period and the main script's prompt does not, so we test both.
2. **Run our version on SciERC with gold pairs.** This tests whether PromptORE finds relations in a scientific domain with unbalanced data.
3. **Add every gold-entity pair.** This measures the effect of pairs without a relation.
4. **Add naming, triples and the graph.** This gives the first complete graph.
5. **Add the filters** (stage 4 and the cluster filter) and measure their effect against milestone 3.
6. **Replace gold entities with GLiNER.** This gives the raw-text pipeline and shows the cost of entity errors.
7. **Test versatility on FRAME**, a very different domain, if time allows.
8. **Write the report** and prepare a short graph demo.

## Main risks

- **Pairs without a relation:** they are the largest share of SciERC pairs and may be hard to filter without labels. This is where the project can add the most to the paper.
- **Small corpora:** clustering needs enough instances. SciERC is much smaller than FewRel, and the elbow rule may be less reliable there.
- **New text:** clusters are computed on a whole corpus. New text would have to be assigned to the nearest cluster, or the corpus clustered again.
- **Misleading names:** a mixed cluster can receive a confident but wrong name, so we review a sample by hand.

## Optional extensions

- Compare with a supervised relation extractor on SciERC.
- Add coreference resolution and measure its effect.
- Link entities to Wikidata.
- Build a small interface to explore the graph.

## To confirm with the teacher

- What are the expected deliverables and deadlines?
