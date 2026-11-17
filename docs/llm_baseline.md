# M4: an LLM baseline for relation extraction with known entities

**Status:** design document, 2026-11-16. Nothing is implemented yet.

This document specifies the fourth method of the plan ([project_plan.md](project_plan.md), section 4,
M4) and of the paper (a generalist reference, on a dev sample). It follows the working rules of
[collaboration.md](collaboration.md): one benchmark, one shared evaluator, every choice made on dev.

The question it answers:

> **Q1 (the plan's first question) is currently answered by two PromptORE variants only.** Both
> use the same `[MASK]` embedding of a masked prompt. A generalist LLM reads the document and the
> pair. Is it better at typing relations than a masked language model trained on the same labels?

The answer is worth having for the paper for two reasons:

1. It tests whether the supervised document model's advantage comes from **reading the whole
   document** rather than from the supervision itself. The current comparison does not separate the
   two: PromptORE also reads less text.
2. It gives the plan's fourth research question a second angle. Q4 asks whether the graph's
   confidence scores can be trusted. An LLM's confidence is a third kind of score (after the
   learned threshold and the logistic probability), and its calibration on this data is a clean,
   cheap result.

---

## 1. Scope and what is deliberately out of scope

**In scope.** A method that produces facts `(doc_id, h, r, t)` with a score, for the same
candidates as every other method (all ordered pairs of different entities of a document), and is
scored by `scripts/evaluate.py`.

**Out of scope.** Entity detection, coreference and entity linking (given by the dataset, as for
every method). Evidence sentence prediction (that is DREEAM's contribution, and it is a separate
method, not a variant of this one). Any use of test labels in the method script.

**Supervision level.** Three settings are specified below, and they land in different tiers of the
leaderboard. The tier decides which published numbers they may be compared with:

| Setting | Uses labels? | `supervision` | Comparable with |
|---|---|---|---|
| S1, end to end | no | `none` | GLiDRE's zero-shot Mistral-Large (18.5 F1) |
| S2, typer on a detected pair set | no (zero-shot) / yes (few-shot, from **train**) | `none` / `supervised` | PromptORE's typing F1 (88.9 for M2, 83.2 for M1) |
| S3, naming the PromptORE clusters | no (the cluster names are LLM-produced) | `labels_for_mapping` | M1 with `labels_for_mapping` (45.9 F1) |

The rule from [collaboration.md](collaboration.md) applies unchanged: numbers from different
supervision levels go in separate rows and are never ranked together. A zero-shot LLM is a
**`none`** run even though it is a strong model; it has simply never seen a label.

**Few-shot examples come from train, never from dev.** Dev labels are for tuning; using them as
in-context examples would leak the tuning set into the method.

---

## 2. Why the LLM is not run end to end on every pair

The naive design (ask the LLM about each of the 193,232 ordered pairs of dev) is not affordable and
not informative. The reason to state explicitly is that it is the *first* thing a reader will ask
about.

dev has 193,232 ordered pairs, of which **93.1% have no relation**. Only 13,362 pairs carry a
relation. Asking a large model about all of them spends 93% of the budget to learn that the answer
is "nothing", and the published zero-shot numbers say the answers are mostly wrong anyway
(Mistral-Large, 18.5 F1 on Re-DocRED test; [GLiDRE](https://arxiv.org/abs/2508.00757), Table 4).

So the three settings below separate the two abilities the project already separates everywhere
else: **detection** (does this pair have a relation?) and **typing** (which one?).

```
S1  end to end      →   detection + typing, one call per document
S2  typer           →   detection from an existing detector, typing by the LLM
S3  cluster naming  →   detection from PromptORE, typing from an LLM-named cluster map
```

S2 is the core experiment. S1 and S3 are cheap and each answers one question of its own.

---

## 3. The three settings

### 3.1 S1: end to end, one call per document

The model receives the document, its entities, and the 96 relation codes, and returns the list of
facts of the document.

- **Calls:** 500 for dev, 100 for dev_debug, 500 for test. Negligible cost.
- **Cost of the design:** the model must emit about 27 facts per document in one answer, and it
  will drop some. This is a *recall* limit of the setting, not a defect of the model, and it is
  exactly what makes the comparison interesting: it shows how much a generalist model loses by not
  being asked one question at a time.
- **What it is for:** the end-to-end reference. It is the row that goes next to GLiDRE's zero-shot
  18.5 F1 in the paper.

### 3.2 S2: the LLM as a typer (core experiment)

The candidate pair set comes from an **existing detector**, so the LLM is only asked to choose a
relation for pairs that are already believed to be related.

Which detector? The best available one is `doc-tune4` (dev detection F1 75.74), or the ensemble
`doc-tune-ens+old`. The pair set is read from the run's saved scores:

```python
# results/<detector>/scores_dev_seed0.pt holds every fact with a score >= 0.02
scores = torch.load("results/doc-tune4/scores_dev_seed0.pt", weights_only=False)
accepted = {k[:3] for k, v in scores.items() if v >= threshold}   # (doc index, h, t)
```

`threshold` is the run's own dev threshold. It is read from the run's training report
(`results/<detector>/train_report_seed<N>.json`, field `best.threshold`), which is where
`train_doc_model.py` writes it. Using the detector's own threshold keeps the two components
consistent.

- **Expected number of calls:** about 12,000 for dev (the detector's detection F1 of 75.74 over
  13,362 gold-related pairs implies roughly 9,600 hits and 2,400 false positives). This number must
  be measured, not assumed, and printed in the run report.
- **What it is for:** the direct answer to Q1. The LLM and PromptORE's typer are compared on
  **the same pair set**, so the only difference is the representation and the model.

**Dependency.** S2 needs a detector's dev scores. They are not in this snapshot of the repository
(`results/` is not versioned and is absent here). The options, in order of preference:

1. Use an existing `results/doc-tune4/` or `results/doc-tune-ens+old/` if it exists on the machine
   that runs the experiment.
2. Produce it once with the documented command (about 50 minutes on the laptop):

   ```bash
   python scripts/train_doc_model.py --run doc-tune4 --tune-layers 4 --pair-features \
       --pair-layers 2 --learning-rate 1e-4 --epochs 15 --seed 0
   ```

3. Run S1 first, which has **no dependency at all** and validates the whole API path, prompt and
   parser. This is the recommended order (see section 8).

### 3.3 S3: the LLM names the PromptORE clusters

PromptORE's M1 clusters the `[MASK]` embeddings of related train pairs, then names each cluster
with the share of the train pairs it contains. The label-free variant of the plan replaces that
naming step with an LLM: for each of the k = 400 clusters, show the model a few member pairs with
their context sentences, and ask which Wikidata property the cluster is about.

- **Calls:** k = 400, one per cluster, plus a second call for the clusters whose answer does not
  match a relation name. Negligible cost.
- **Output:** a mapping `cluster id -> relation code`, saved as `results/<run>/cluster_names.json`
  and committed, because it is part of the method and not a prediction.
- **Mapping the answer back:** the model answers with a relation name in natural language; it is
  matched against `rel_info.json` (code -> name), and every cluster whose match is ambiguous or
  empty is listed in `cluster_names.json` under `unmatched` for manual assignment. This manual step
  must be recorded, because it is a human decision made with the labels in view.
- **What it is for:** the cleanest ablation of M1. M1 and S3 differ in **one step only** — how the
  clusters are named — and they are in the same tier of the leaderboard (`labels_for_mapping`),
  because the PromptORE detector is still trained on the train labels in both. The question it
  answers is "how much of M1's performance is the clustering, and how much is the naming?".

  Note that a genuinely label-free run would need a label-free detector as well, which the project
  does not have. The plan's "label-free variant" of M1 (section 4) removes the labels from the
  naming step, which is exactly what S3 does; the detector still needs them. If the team wants a
  fully unsupervised run, the cheapest label-free detector is the `[MASK]` log-probability of the
  relation names at the mask position, and that is a separate, optional experiment.

---

## 4. Inputs: what the model is shown

The dataset already gives everything; nothing has to be engineered. Two helpers of the repository
are reused as they are:

| Input | Source |
|---|---|
| Document text, per sentence | `doc["sents"]` |
| Entity list with type, longest mention and all mentions | `doc["entities"]` |
| Entity display name | `relations.prompts.entity_name(entity)` (longest mention) |
| Full-document context for a pair | `relations.prompts.build_context(doc, head, tail, "document")` |
| The 96 relation codes and names | `rel_info.json` (`data.linked_docred.load_relation_names`) |
| Relation frequency (for the label space and for reporting) | `data.redocred.relation_buckets` and the counts of the train split |

`build_context` already supports the `"document"` context, so the S2 prompt needs no new context
code. The PromptORE settings use `"mention_sentences"`, which is what makes the two methods differ
in how much text they read (one sentence pair against the whole document).

### 4.1 The entity table

Entities are referred to by their **position in `doc["entities"]`**, never by their name alone:
the model has to output a pair that the evaluator can map back to entity indices, and names are
ambiguous and are sometimes reworded by the model.

```
0 | PER | Willi Schneider | mentions: "Willi Schneider" (s0), "Schneider" (s3)
1 | LOC | Mediaș           | mentions: "Mediaș" (s0)
```

The mention list is included because a relation is usually stated in one particular mention, and
because it tells the model which sentences matter.

### 4.2 The relation table

All 96 codes are given, with their English names, because the names carry the semantics
(`P17 country`, `P19 place of birth`). The model chooses among them; it never invents a code.

```
P17  country
P19  place of birth
P131 located in the administrative territorial entity
...
```

The table is about 1,150 tokens and is identical in every call of a setting. It goes in the system
prompt, so a provider with prompt caching pays for it once per run rather than per call.

---

## 5. Prompts

### 5.1 S2, the typer (the prompt that matters)

```text
SYSTEM:
You extract relations from a Wikipedia introduction. The entities are already known and
numbered. You are given one ordered pair of entities and you list the Wikidata properties
that relate the head to the tail.

Most entity pairs in this corpus have NO relation: in the training set, 93% of them have
none. Return an empty list when the document does not state a relation between the two
entities. Do not guess from world knowledge: only relations supported by this document count.

Rules:
- Use only the relation codes of the list below.
- A pair may have several relations; a relation may also be symmetric, in which case list it
  once for the given direction.
- Answer with the requested format and nothing else.

USER:
[RELATIONS]
P17 country
P19 place of birth
... (96 lines)

[DOCUMENT]
Title: Willi Schneider (skeleton racer)
0: Willi " Willi " Schneider ( born 13 March 1963 in Mediaș , Romania ) was a Romanian
   skeleton racer .
1: He competed at the 1988 Winter Olympics in Calgary .

[ENTITIES]
0 | PER | Willi Schneider | mentions: "Willi Schneider" (s0), "Schneider" (s0)
1 | TIME | 13 March 1963 | mentions: "13 March 1963" (s0)
2 | LOC | Mediaș | mentions: "Mediaș" (s0)
3 | LOC | Romania | mentions: "Romania" (s0)
4 | LOC | Calgary | mentions: "Calgary" (s1)

[QUESTION]
Which relations hold from entity 0 (head) to entity 3 (tail)?

[FORMAT]
RELATIONS: P17, P131
CONFIDENCE: 0.85

RELATIONS is a comma-separated list of codes, or NONE if the document states no relation
between the head and the tail. CONFIDENCE is a number between 0 and 1.
```

Two design points worth keeping:

- **The 93% prior is stated.** Without it, a generalist model over-predicts, and the run measures
  the model's willingness to say "nothing" rather than its typing ability.
- **The answer is a plain two-line format, not JSON.** It parses without a JSON parser, and it
  leaves room for the confidence field, which the evaluator needs for the graph metrics (see
  section 6.3).

### 5.2 S1, end to end

Same document, entity table and relation table, but the question is "list every relation of every
pair of this document", and the expected answer is a list of `head tail relation` lines.

```
[FORMAT]
FACT: 0 P17 3
FACT: 0 P19 1
```

The model is told that it may return an empty list, and that a document has about 27 true
relations. The recall limit of this format is a property of the setting and is reported as such.

### 5.3 S3, cluster naming

For each cluster, a list of member pairs, each with the sentence that links them:

```text
[CLUSTER 137]
members (pair | sentence):
  (Mediaș, Romania) | "Willi Schneider was a Romanian skeleton racer from Mediaș."
  (Mediaș, Romania) | "He was born in Mediaș, Romania, in 1963."
  ...

[QUESTION]
Which Wikidata property of the list above do these pairs share?
Answer with one code and nothing else.
```

### 5.4 Few-shot variant (optional, S2 only)

Four to sixteen examples drawn from **train**, each a short document fragment with the relation
that holds. They are fixed once and stored in `scripts/llm_fewshot.py` so that every run uses the
same ones. This variant uses labels, so it is `supervised` and goes in its own row.

---

## 6. Producing the prediction files

### 6.1 The two output formats

| Setting | Prediction file |
|---|---|
| S1 | one row per fact the model states: `doc_id,h,r,t,score` |
| S2 | one row per fact the model states for a candidate pair; the candidate set is the detector's |
| S3 | the PromptORE facts, with the relation of each pair replaced by the LLM-named cluster relation |

All three are ordinary run folders for the shared evaluator, and all three are written with
`evaluation.runs.write_shared_run`, which handles the `dev_debug` filtering and the `card.json`:

```python
from kg_extraction.evaluation.runs import write_shared_run
write_shared_run(Path("results/llm-gpt4omini"), docs, facts, card, seed)
```

`facts` maps `"dev"` (and optionally `"test"`) to `{(doc_index, head, tail, relation): score}`.
**The key order is `(index, h, t, r)`**, the one used by `evaluation/relations.py` and by
`graph/build.py`. The shared evaluator uses `(doc_id, h, r, t)` internally; it converts. Mixing the
two orders gives a zero score with no error message.

### 6.2 `card.json`

```json
{
  "run": "llm-gpt4omini",
  "author": "<author>",
  "method": "Generalist LLM as typer on the pairs of doc-tune4",
  "supervision": "none",
  "training_data": "none",
  "models": "gpt-4o-mini (2024-07-18), temperature 0",
  "description": "One call per accepted pair, full document as context, 96 relation codes; \
                  fact confidence = mean token log-probability of the answer"
}
```

[collaboration.md](collaboration.md) requires the exact model name and the date in the card, and
temperature 0. The date matters: API models change under the same name, so a run that cannot be
reproduced is still valuable as a dated measurement, but must be labelled as such.

### 6.3 The `score` column

The evaluator needs a score for the graph metrics (`graph_precision_at_N`,
`graph_precision_support_*`). Without a `score` column, those columns are empty for the run.

Three ways to get one, in order of preference:

1. **Token log-probabilities.** Ask the provider for `logprobs` and take the geometric mean of the
   probabilities of the answer tokens. Objective, free, and it is a real confidence measure.
2. **Self-consistency.** Run the same call three times at temperature 0.7 and use the share of
   runs that produced the same answer. Costs three times as much; it is the fallback if the
   provider does not return log-probabilities.
3. **The model's own `CONFIDENCE` field.** Cheapest, and the least reliable: the paper already
   reports that LLM self-reported confidence is poorly calibrated.

The S2 fact score is the LLM's. The detector's own score for the same pair is kept in the raw
response file, so that the combined score (detector x LLM) can be compared as an ablation.

### 6.4 Raw answers

[collaboration.md](collaboration.md), rule 5: keep the raw answers, because API models change over
time. Every call is written to `results/<run>/raw/<subset>/<doc_id>_<h>_<t>.json` with the prompt,
the raw completion, the token counts and the timestamp. These files are not versioned (they are
large) but the run report and the card are.

---

## 7. Cost

The token counts below are estimates from the sizes of the inputs (a document is about 600
tokens, the entity table about 250, the relation table about 1,150) and must be replaced by the
measured numbers in Phase 2. Prices change; the provider's price list at the time of the run goes
into the run report.

| Setting | Calls (dev) | Input tokens | Output tokens |
|---|---|---|---|
| S1 | 500 | about 1.0 M | about 0.4 M |
| S2 | about 12,000 | about 26 M | about 0.4 M |
| S2, dev_debug only | about 2,400 | about 5 M | about 0.1 M |
| S3 | 400 | about 0.4 M | about 0.05 M |

S2 is the only setting whose cost is not negligible, and it is the one that matters. Two levers
keep it small:

- **Run the main model on dev only**, and only promote the chosen configuration to test. Test is
  scored once, at the end, with frozen settings (rule 1 of the plan, section 6).
- **Use a small model for the full run and a large model for a subset.** A full dev run with a
  small model, plus a `dev_debug` run with a large one, gives the ordering at a fraction of the
  cost.

The run script prints, for every call, the number of input and output tokens, and writes the
totals to `results/<run>/llm_report.json`, so the cost of a configuration is known before it is
run at full scale.

---

## 8. Execution plan

Each phase has an acceptance condition. The phases are ordered so that the cheapest one runs
first, and so that no phase depends on artefacts that do not exist yet.

### Phase 0: environment

- Add the provider to the API configuration. On this machine, `C:/Users/lenovo/.minimax/config.yaml`
  currently holds only the `minimax` provider (`MiniMax-M3`, `MiniMax-M3.1-Flash`, an
  OpenAI-compatible base URL). An `openai` or `deepseek` provider is added as a sibling of
  `minimax` under `provider:`, with its own key.
- **The key never enters the repository.** `scripts/llm_relations.py` reads it from an environment
  variable (`LLM_API_KEY`, or `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` per provider) and the
  repository stores only the provider name, the model id and the base URL.
- Add `openai` to `requirements.txt`. Note that `tqdm` is imported by `relations/doc_encoder.py`
  and `relations/encoder.py` but is missing from `requirements.txt` today; it only installs
  because `transformers` depends on it.

*Acceptance:* one call returns a valid answer, and the token count is printed.

### Phase 1: S1 on dev_debug (no dependencies)

- Implement the client (retry with backoff, rate limit, per-call JSON log, token accounting).
- Implement the prompt of section 5.2 and a strict parser: a malformed line is logged and counted,
  never silently dropped.
- Run on the 100 documents of `dev_debug`.
- **Read 30 of the answers by hand.** This is the step that decides whether the prompt needs work;
  it is cheap and it catches the failure modes that a score alone hides (invented relations,
  wrong direction, answers on the wrong pair, empty answers).

*Acceptance:* every call is logged, the parse failure rate is below 2%, and a manual read of 30
answers shows no systematic format problem.

### Phase 2: S1 on dev, and the detector

- Run S1 on the 500 documents of dev, produce the run folder, score it with the shared evaluator.
- In parallel, produce the detector scores of section 3.2 if they are not already on the machine.

*Acceptance:* `scripts/evaluate.py results/llm-<model>` prints a full metric row, and the number
matches the order of magnitude of the published zero-shot results (GLiDRE, 18.5 F1). A large gap
in either direction means the prompt or the parsing is wrong, not that the model is better or
worse.

### Phase 3: S2 on dev_debug

- Run the typer on the accepted pairs of `dev_debug` (about 2,400 calls).
- Score it with the shared evaluator, on the same subset, and compare with the detector's own run
  on the same subset.

*Acceptance:* a typing F1 and a micro F1, and a manual read of 20 answers in which the model
rejected a pair (it must be able to say NONE).

### Phase 4: S2 on dev

- About 12,000 calls. The run is resumable: the per-call log is the state, and a restart skips
  the calls that are already there.
- Score it, and run the paired bootstrap against `doc-tune4` and against `doc-tune-ens+old`:

  ```bash
  python scripts/evaluate.py results/llm-<model> --leaderboard report/leaderboard.csv
  python scripts/compare_runs.py results/llm-<model> results/doc-tune4 --subset dev
  ```

*Acceptance:* a complete row in `report/leaderboard.md`, with a 95% interval and a p-value against
the document model.

### Phase 5: S3, and the ablations

- S3 on the k = 400 clusters, with the manual mapping of the unmatched clusters.
- The ablation grid of section 9.

*Acceptance:* the cluster naming is committed as `cluster_names.json`, with the unmatched clusters
and the manual assignments recorded.

### Phase 6: test, once

Only after every choice above has been made on dev. The run folder gets `"test_status": "final"`,
and the test predictions are produced by one final pass with the frozen settings.

---

## 9. Ablations

Each row is a configuration of the same method; each is chosen on dev.

| Dimension | Values |
|---|---|
| Model | the main model, plus one or two others |
| Setting | S1, S2, S3 |
| Context | whole document / the two closest mention sentences / the shared sentence |
| Relation space | all 96 codes / the 10 most frequent codes only / free-form names mapped afterwards |
| Few-shot examples | 0 / 4 / 16 (from train) |
| Answer format | the two-line format / JSON |
| Temperature | 0 (the default) / 0.7 x 3 votes, for the self-consistency score |

The context row is the most informative, because it is the direct test of the hypothesis in
section 1: **the advantage of the document model should come from the context.** If the LLM loses
most of its accuracy when the context shrinks to two sentences, that is the same effect that
separates PromptORE from the document model, and it is the sentence that closes the paper's
argument.

The relation-space row answers "is the LLM's advantage only that it knows the label space?".

---

## 10. Metrics, and one caveat about them

Everything is scored by `scripts/evaluate.py`, with the notebook version
`notebooks/redocred_eval_cell.py` kept in sync (rule 3 of [collaboration.md](collaboration.md)).

**The caveat.** `typing_scores` keeps only the pairs that are related in the gold annotations
**and** predicted related. A pair that the detector accepted, that is related in gold, and that the
LLM rejected with `NONE` is therefore **not** counted as a typing error: it disappears from the
typing metrics, and shows up only as a recall loss in the overall F1.

That is correct for the metric as defined, but it means a typer that plays safe is flattered by
the typing F1. The report must therefore give, next to the standard typing F1, the following two
numbers for the detector's accepted pairs:

- **kept rate**: of the accepted pairs that are related in gold, the share the LLM kept;
- **right rate**: of the pairs that are related in gold and kept, the share where the relation is
  right.

These two together say what the typing F1 alone does not: whether the LLM is genuinely better at
choosing relations, or merely more willing to answer.

Where do these two numbers live? The rule of [collaboration.md](collaboration.md) says that a new
metric goes into the shared evaluator, and that shared code changes only in its own pull request,
reviewed by another member. If the team wants these two columns in the leaderboard, they go into
`scripts/evaluate.py` **and** `notebooks/redocred_eval_cell.py`, in a separate change. If not,
they are reported in the run's own `llm_report.json` and in the paper's analysis section, and are
not leaderboard columns. **This is a decision for the team, not for the author of this method.**

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| The API model changes under the same name | The exact model id and the date go in `card.json`; the raw answers are kept; the date is in the paper's table |
| The cost of S2 on dev is higher than expected | Phase 3 measures the cost on `dev_debug` first; the run is resumable; the budget is a hard stop in the script (`--max-calls`, `--max-tokens`) |
| The answers do not parse | Strict parser, failures counted and logged, one retry with a stricter instruction; the parse failure rate is in the run report |
| The model invents relations | The answer must use codes of the list; a code outside the list is dropped and counted; the counter is reported |
| The model answers about the wrong pair | The prompt asks about one pair with its indices; the answer is checked against the pair that was asked |
| **The annotations are incomplete**, so true facts are scored as errors | This is a known limit of the dataset (risk 5 of the plan, section 9). The same limit applies to every method, so the comparison stays fair; a sample of the LLM's false positives is checked by hand, and the finding is reported |
| The zero-shot LLM is judged against a supervised method | It is not: `supervision: none` goes in its own row, and the comparison is made against PromptORE's zero-shot and the published zero-shot numbers |
| A dev label leaks through a few-shot example | Examples come from train only; a unit test asserts that no example comes from dev or test |

---

## 12. Files

New files:

```text
scripts/llm_relations.py                     the CLI: settings S1, S2, S3
scripts/llm_fewshot.py                       the fixed few-shot examples, from train
src/kg_extraction/relations/llm_client.py    the API client: retry, rate limit, log, token count
src/kg_extraction/relations/llm_prompts.py   the prompts and the parsers of section 5
results/<run>/card.json                      written by write_shared_run
results/<run>/llm_report.json                settings, token counts, parse failures, timings
results/<run>/cluster_names.json             S3 only: the LLM naming of the clusters
results/<run>/raw/                           the raw answers (not versioned)
docs/llm_baseline.md                         this document
```

Changed files:

- `requirements.txt`: add the API client library.
- `report/leaderboard.csv` and `report/leaderboard.md`: written by `scripts/evaluate.py`.
- `report/paper/main.tex`: the new section and the new rows of the results tables.
- `docs/project_plan.md`: the status of step 11 (LLM), and the M4 row of section 4.

Unchanged, on purpose: `src/kg_extraction/evaluation/*` (shared code), unless the team decides on
the two extra metrics of section 10.

---

## 13. Acceptance criteria

The method is done when:

1. `scripts/llm_relations.py` runs S1, S2 and S3 from a clean checkout, with the API key read
   from the environment.
2. Every run folder passes `scripts/evaluate.py` and produces a row in `report/leaderboard.md`
   with the same metrics as every other method.
3. `scripts/compare_runs.py` gives a paired bootstrap against `doc-tune4` on dev.
4. The raw answers of every call are kept, and the token counts of every configuration are in the
   run report.
5. The model id and the date of every run are in its `card.json`.
6. The two extra numbers of section 10 (kept rate, right rate) are reported next to the typing
   F1, wherever the team decides to put them.
7. The paper has a section that states the answer to Q1 in one paragraph, with the numbers.

## 14. Open questions for the team

1. **Which model?** The local configuration has MiniMax only. The paper's published zero-shot
   reference is Mistral-Large (GLiDRE). To quote a number comparable to the published ones, the
   main model should be one of those: Mistral-Large, GPT-4o, DeepSeek, or Llama, all through an
   API.
2. **Which setting is the paper's "LLM" row?** S1 (end to end, comparable to the published
   zero-shot numbers) and S2 (typer, comparable to PromptORE's typing) answer different questions
   and should probably both be in the paper, in different rows of different tables.
3. **Is a zero-shot LLM enough, or is a few-shot variant needed?** The few-shot variant uses
   labels, so it is a `supervised` run, and it belongs next to the supervised document model
   rather than next to PromptORE.
4. **The two extra metrics of section 10: leaderboard columns, or analysis only?** This changes
   whether `scripts/evaluate.py` is touched.
