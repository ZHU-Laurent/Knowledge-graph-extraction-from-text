# Dataset selection

## Selected datasets

| Role | Dataset | Content | Access | License |
|---|---|---|---|---|
| Main dataset | Linked-Re-DocRED | Wikipedia documents with gold entities, coreference, entity links, 96 relations and a labeled test set | [Linked-DocRED repository](https://github.com/alteca/Linked-DocRED/tree/main/Linked-Re-DocRED) | GPL-3.0 |
| Reproduce the PromptORE paper | [FewRel](https://github.com/thunlp/FewRel) | Wikipedia, 56,000 instances, 80 relations. Variants: FewRel NYT (2,500 instances, 25 relations) and FewRel PubMed (1,000 instances, 10 relations) | GitHub | MIT |

## Linked-DocRED or Linked-Re-DocRED

| Measure | Linked-DocRED | Linked-Re-DocRED |
|---|---|---|
| Documents (train, dev, test) | 3,053, 998, 1,000 | 3,053, 500, 500 |
| Relations in train | 38,058 | 85,932 |
| Test set relation labels | **None** | **17,448** |

The Linked-DocRED test set withholds relation labels, so we could not score on it ourselves. DocRED's annotations are also known to be incomplete, which is a problem when "no relation" is one of our classes.

## Linked-Re-DocRED measurements

| Measure | Train | Dev | Test |
|---|---|---|---|
| Entities | 59,359 | 9,684 | 9,779 |
| Relations | 85,932 | 17,284 | 17,448 |
| Relation types present | 96 | 95 | 95 |
| Ordered entity pairs | 1,193,092 | 193,232 | 198,670 |
| Ordered pairs with a relation | 67,808 (5.7%) | 13,362 (6.9%) | 13,627 (6.9%) |
| Related pairs with several labels | 25.6% | 27.3% | 26.6% |
| Related pairs also related in reverse | 39.8% | 42.1% | 40.9% |
| Related pairs sharing a sentence | 48.3% | 47.5% | 47.6% |
| Relation types with fewer than 50 instances | 5 | 30 | 33 |
| Share of the 10 largest relation types | 64.0% | 60.7% | 62.1% |

The largest relations in train are "located in the administrative territorial entity" (23.7%), "country" (16.8%) and "country of citizenship" (5.4%).

Each document has a title, tokenized sentences, entities and relations. Each entity has an id, a type (`NUM`, `TIME`, `PER`, `ORG`, `LOC` or `MISC`), its mentions and an entity link. Each relation has a Wikidata code `r`, a head `h`, a tail `t` and evidence sentences.

## Why FewRel alone is not enough

FewRel annotates one entity pair per instance, has at most one instance per pair and has no "no relation" class. We only use it to check that we can reproduce the paper.

## Datasets set aside

- **SciERC, FRAME, SciER:** useful for the earlier end-to-end pipeline or other domains, but not needed for the reduced scope. They can serve as an extension.
- **Linked-DocRED:** its test set has no relation labels.
- **CoNLL04:** too few relations for a meaningful clustering study.
- **TACRED, Re-TACRED:** access may require a license.
- **REBEL, SREDFM, T-REx:** automatic annotation, less reliable for evaluation.
