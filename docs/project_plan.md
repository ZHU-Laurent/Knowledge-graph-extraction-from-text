# Knowledge graph project: proposed plan

**Scope:** extract entities and explicit relations from text and turn them into a knowledge graph. We start from the PromptORE baseline and its FewRel dataset, then move to the SciERC dataset (about 500 AI paper abstracts with annotated entities, relations and coreference).

**Research question:** which extraction approach produces the best graph, at what cost, and with what kinds of errors?

## Baseline: PromptORE

Our baseline is [PromptORE](https://arxiv.org/abs/2304.01209) (Genest et al.), an unsupervised relation extraction method. It uses a prompt to embed each entity pair in its sentence, then clusters these embeddings to discover candidate relations. The labels are only used for evaluation.

**Dataset:** we first use FewRel, the main dataset of the paper (56,000 Wikipedia instances, 80 relations). FewRel NYT and FewRel PubMed are the paper's two smaller, domain-specific datasets.

1. **Get the code and data:** use the [PromptORE code](https://github.com/alteca/PromptORE) (GPL-3.0, tested on Python 3.8) and the [FewRel data](https://github.com/thunlp/FewRel).
2. **Reproduce the result:** run PromptORE on FewRel with the number of relations set to 80. With BERT, the paper reports a B³ F1 of 48.8, a V-measure F1 of 71.8 and an ARI of 43.4.
3. **Document any differences:** record our scores, the seed and the settings, and explain any gap with the paper.

PromptORE uses clustering metrics and receives the entity pairs as input. It does not detect entities or name its clusters, so comparing it with the next phase requires a clear protocol.

## Next phase: full pipeline

1. **Set up the data:** load SciERC, keep the official train, dev and test splits, and leave the test set untouched until the final evaluation.
2. **Build the evaluation first:** count a relation as correct only if both entities, their types and the relation label all match. Compute precision, recall and F1.
3. **Build the entity model:** fine-tune a Transformer to detect and classify entities such as Method, Task, Metric and Material.
4. **Build a rule-based relation extractor:** use text and syntax patterns, then run the pipeline end to end. This is the simplest extractor in this phase.
5. **Add two more relation extractors:** fine-tune a Transformer classifier on entity pairs, and prompt an LLM with the allowed relations, a few examples and JSON output. Reject anything outside the schema.
6. **Normalize the entities:** merge mentions of the same thing (CNN, CNNs, convolutional neural network) into one node, starting with simple rules.
7. **Build the graph:** store entities as nodes and relations as edges, with the source sentence on each edge. Query and visualize it with NetworkX or Neo4j.
8. **Compare and analyze:** score each extractor on gold entities and on predicted entities, to see how much entity errors cost. Break errors down by relation type, and measure invented relations for the LLM.
9. **Write up the results:** produce the report and a short graph demo.

## Optional extensions

- Add coreference resolution and measure its impact.
- Link entities to Wikidata.
- Run the pipeline on new arXiv abstracts.
- Build a small interface to explore the graph.

## Justifying our choices

We are free to choose our resources, as long as we justify each choice. The report should explain the following decisions, and the team can still change them.

| Choice                                     | Why                                                                                                     | Alternatives set aside                                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| SciERC dataset                             | Small and consistent, with annotated entities, relations and coreference, so we can evaluate every step | CoNLL04: gives a poor graph. DocRED: heavier, with incomplete annotations. TACRED: requires a license |
| Rules, a fine-tuned Transformer and an LLM | They represent the three main families of methods, so we can compare accuracy, cost and errors          | A single method: leaves nothing to compare                                                            |
| Strict relation matching                   | Standard criterion that avoids overstating results, a known problem in this field                       | Relaxed matching: can inflate scores, so we would only report it in addition                          |
| Scoring on gold and predicted entities     | Separates relation errors from errors passed on by the entity model                                     | Predicted entities only: hides where the errors come from                                             |
| Simple rules first for normalization       | Easy to explain and inspect, and gives a baseline for any later improvement                             | Wikidata linking: kept as an extension because it adds work                                           |

For each choice, we should keep the supporting evidence: the paper it relies on, or our own measurements.

## To confirm with the teacher

- What are the expected deliverables and deadlines?
