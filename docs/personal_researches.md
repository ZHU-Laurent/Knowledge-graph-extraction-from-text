# Dataset selection

Only the FewRel files and an unofficial copy of SciERC have been inspected so far. Check the formats, annotations and licenses of the others before use.

## Selected datasets

| Role | Dataset | Why | Content | Access | License |
|---|---|---|---|---|---|
| Reproduce the baseline | [FewRel](https://github.com/thunlp/FewRel) | The dataset of the [PromptORE](https://arxiv.org/abs/2304.01209) paper | Wikipedia, 56,000 instances, 80 relations. Variants: FewRel NYT (25 relations, 2,500 instances) and FewRel PubMed (10 relations, 1,000 instances) | GitHub, also on [Hugging Face](https://huggingface.co/datasets/thunlp/few_rel) | MIT |
| Evaluate the pipeline | [SciERC](http://nlp.cs.washington.edu/sciIE/) | Annotates every stage: entities, all relations in each text and coreference | 500 AI paper abstracts, 6 entity types, 7 relations | Official site or the version in the [PURE](https://github.com/princeton-nlp/PURE) repository | Unknown |
| Test versatility (if time allows) | [FRAME](https://arxiv.org/abs/2602.19133) | A very different domain, built for knowledge graph construction | Art-historical descriptions, 37 entity types, typed relations, coreference, Wikidata alignment | [Zenodo](https://doi.org/10.5281/zenodo.18724632) | CC BY 4.0 |
| Test longer documents (optional) | [SciER](https://arxiv.org/abs/2410.21155) | Full papers rather than abstracts | 106 AI papers, over 24,000 entities, over 12,000 relations | [GitHub](https://github.com/edzq/SciER) | Unknown |

## Why FewRel is not enough

FewRel annotates only one entity pair per instance and has at most one instance per pair. It cannot evaluate entity detection or pairs without a relation, and it would produce a sparse graph. We use it to reproduce the paper, not to evaluate the whole pipeline.

## SciERC measurements

Measured on the unofficial copy [zj88zj/SCIERC](https://huggingface.co/datasets/zj88zj/SCIERC). Measure again on the official files.

| Measure | Value |
|---|---|
| Documents (train, dev, test) | 350, 50, 100 |
| Sentences | 2,687, the longest with 101 tokens |
| Entities | 8,094, including 1,338 `Generic` |
| Relations | 4,648, none across sentences |
| Same-sentence entity pairs | 12,211, of which 4,644 (38%) have a relation |

| Relation | Share |
|---|---|
| `USED-FOR` | 52.4% |
| `CONJUNCTION` | 12.5% |
| `EVALUATE-FOR` | 9.8% |
| `HYPONYM-OF` | 8.8% |
| `PART-OF` | 5.8% |
| `FEATURE-OF` | 5.7% |
| `COMPARE` | 5.0% |

## Points to check

- **SciERC copies:** copies on Hugging Face are unofficial and state no license. Use the official source for final results.
- **FRAME:** check the number of documents and relations on Zenodo, and test the conversion from its UIMA XMI format.

## Datasets set aside

- **CoNLL04:** SciERC is already small, and FewRel PubMed covers quick tests of the relation step.
- **DocRED, Re-DocRED, DWIE:** useful for relations across sentences, which our pipeline does not handle.
- **TACRED, Re-TACRED:** access may require a license.
- **REBEL, SREDFM, T-REx:** automatic annotation, less reliable for evaluation.
- **POPCORN, REDFM:** only relevant if the project moves to French.
