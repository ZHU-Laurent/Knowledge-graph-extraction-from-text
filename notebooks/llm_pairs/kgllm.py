"""LLM pair representations inside the PromptORE pipeline of the team repository (methods M1 and M2).

Everything except the representation of a pair is the repository's own code and protocol:
candidates, context ("mention_sentences"), entity names, train sample (all related pairs + 3 unrelated
per related, seed 0, weighted), logistic detector, one logistic classifier per relation (M2), k-means
clusters named by the share of train relations (M1), fact score P(related) x typing score, fact
threshold and k chosen on dev, shared evaluator (scripts/evaluate.py).

Representations of a pair:
  bert        BERT [MASK] state: the repository's PromptORE, unchanged (reference)
  qwen-state  state of a causal LLM at the position where its answer starts         ("method 1")
  qwen-text   the LLM answers "relation | head type | tail type" -> sentence encoder ("method 2")
  api-text    same prompt answered by an API model (DeepSeek)                       ("method 2.5")
  bert+text   bert and the best text representation, concatenated
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import threading
import time

import numpy as np

SEP = "\x1f"


# ============================================================================ resumable cache
class Cache:
    """Runs fn over items in length-sorted chunks and saves each chunk, so a restarted run resumes."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def run(self, name, items, fn, chunk=2048, kind="npy"):
        """kind: "npy" (fn -> array), "json" (fn -> list) or "both" (fn -> (array, list))."""
        if kind == "both":
            return self._both(name, items, fn, chunk)
        return self._run(name, items, fn, chunk, kind)

    def _both(self, name, items, fn, chunk):
        """Chunk by chunk: the array first, then the texts of the same chunk."""
        sig = hashlib.sha1("\x1e".join(items).encode("utf-8")).hexdigest()[:10]
        folder = os.path.join(self.root, f"{name}_{sig}")
        os.makedirs(folder, exist_ok=True)
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        arrays, texts, t0, done = [None] * len(items), [None] * len(items), time.time(), 0
        n_chunks = (len(order) + chunk - 1) // chunk
        for c in range(n_chunks):
            idx = order[c * chunk:(c + 1) * chunk]
            a_path, t_path = (os.path.join(folder, f"part{c:04d}.{ext}") for ext in ("npy", "json"))
            a = t = None
            if os.path.exists(a_path) and os.path.exists(t_path):
                try:
                    a, t = np.load(a_path).astype(np.float32), json.load(open(t_path, encoding="utf-8"))
                except Exception:
                    a = t = None
                if a is not None and (len(a) != len(idx) or len(t) != len(idx)):
                    a = t = None
            if a is None:
                batch = [items[i] for i in idx]
                a, t = fn(batch)
                a, t = np.asarray(a), list(t)
                if len(a) != len(idx) or len(t) != len(idx):
                    raise ValueError(f"{name}: wrong number of results")
                np.save(a_path + ".tmp.npy", a.astype(np.float16))
                with open(t_path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(t, f)
                os.replace(t_path + ".tmp", t_path)
                os.replace(a_path + ".tmp.npy", a_path)
                a = a.astype(np.float16).astype(np.float32)
                done += len(idx)
                rate = done / max(time.time() - t0, 1e-6)
                left = len(order) - (c + 1) * chunk
                print(f"  {name}: chunk {c + 1}/{n_chunks}  {rate:.0f} items/s  about "
                      f"{max(left, 0) / max(rate, 1e-6) / 60:.0f} min left", flush=True)
            for i, row, text in zip(idx, a, t):
                arrays[i], texts[i] = row, text
        return np.stack(arrays).astype(np.float32), texts

    def _run(self, name, items, fn, chunk=2048, kind="npy"):
        sig = hashlib.sha1("\x1e".join(items).encode("utf-8")).hexdigest()[:10]
        folder = os.path.join(self.root, f"{name}_{sig}")
        os.makedirs(folder, exist_ok=True)
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        out, t0, done = [None] * len(items), time.time(), 0
        n_chunks = (len(order) + chunk - 1) // chunk
        for c in range(n_chunks):
            idx = order[c * chunk:(c + 1) * chunk]
            path = os.path.join(folder, f"part{c:04d}.{kind}")
            res = None
            if os.path.exists(path):
                try:
                    res = np.load(path).astype(np.float32) if kind == "npy" else json.load(open(path, encoding="utf-8"))
                except Exception:
                    res = None
                if res is not None and len(res) != len(idx):
                    res = None
            if res is None:
                res = fn([items[i] for i in idx])
                if len(res) != len(idx):
                    raise ValueError(f"{name}: {len(res)} results for {len(idx)} items")
                tmp = os.path.join(folder, f"tmp{c:04d}.{kind}")
                if kind == "npy":
                    np.save(tmp, np.asarray(res).astype(np.float16))
                    res = np.asarray(res).astype(np.float16).astype(np.float32)
                else:
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(list(res), f)
                os.replace(tmp, path)
                done += len(idx)
                rate = done / max(time.time() - t0, 1e-6)
                left = sum(len(order[k * chunk:(k + 1) * chunk]) for k in range(c + 1, n_chunks))
                print(f"  {name}: chunk {c + 1}/{n_chunks}  {rate:.0f} items/s  about {left / max(rate, 1e-6) / 60:.0f} min left",
                      flush=True)
            for i, r in zip(idx, res):
                out[i] = r
        return np.stack(out).astype(np.float32) if kind == "npy" else out


def inst_key(inst):
    return SEP.join((inst["context"], inst["head"], inst["tail"]))


def key_inst(key):
    context, head, tail = key.split(SEP)
    return {"context": context, "head": head, "tail": tail}


# ============================================================================ local causal LLM
STRUCT_SYSTEM = (
    'You label the relation between two entities mentioned in a text.\n'
    'Answer with one line, in exactly this format and with nothing else:\n'
    '<relation> | <head type> | <tail type>\n'
    '<relation> is a short phrase (1 to 5 words) for the relation read from the head entity to the tail '
    'entity, or the word none when the text states no relation from the head entity to the tail entity. '
    'Many pairs of entities have no relation.\n'
    '<head type> and <tail type> are the general types of the two entities (1 to 3 words each).\n'
    'Use only what the text says.')
STRUCT_USER = 'Text: "{context}"\nHead entity: "{head}"\nTail entity: "{tail}"'


def struct_user(inst):
    return STRUCT_USER.format(**inst)


def _oom(e):
    return "out of memory" in str(e).lower()


class CausalLM:
    """One loaded LLM for the last-token embeddings (C) and the generated answers (D2)."""

    def __init__(self, model_name, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"            # the last position is the last real token
        self.tok.truncation_side = "left"         # never cut the end of the prompt
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        self.device = device

    def _batched(self, texts, bs, step):
        """Apply step to batches; on a GPU out-of-memory error the batch size is halved and retried."""
        out, i = [], 0
        while i < len(texts):
            try:
                out.extend(step(texts[i:i + bs]))
                i += bs
            except RuntimeError as e:
                if not _oom(e) or bs == 1:
                    raise
                self.torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"  out of memory, batch size -> {bs}", flush=True)
        return out

    def answer_with_state(self, users, system=None, bs=32, max_new_tokens=24, max_len=768):
        """One pass per prompt: the generated answer, and the last-layer state at the last token of the
        prompt, the position from which the model starts to write its answer.

        -> (states (n, hidden), answers). The state is read at the input of the output layer during the
        first step of the generation, so the prompt is only encoded once."""
        torch = self.torch
        prompts = []
        for u in users:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]
            prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        captured = []

        def hook(module, args):
            if not captured:                        # first call = the whole prompt
                captured.append(args[0][:, -1, :].detach().float().cpu().numpy())
        handle = self.model.get_output_embeddings().register_forward_pre_hook(hook)

        def step(batch):
            captured.clear()
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_len).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if not captured or len(captured[0]) != len(batch):
                raise ValueError("the state of the prompt was not captured")
            return list(zip(captured[0], texts))
        try:
            out = self._batched(prompts, bs, step)
        finally:
            handle.remove()
        return np.stack([state for state, _ in out]), [text for _, text in out]


# ============================================================================ parsing the answers
def clean_phrase(p):
    p = p.strip().split("\n")[0].strip().strip('"\'`.:;,*<> ').strip().lower()
    return " ".join(re.sub(r"\s+", " ", p).split()[:6])


NONE_WORDS = {"", "none", "no relation", "n/a", "na", "null", "unknown", "no", "nothing", "not stated",
              "no relationship", "none stated", "unrelated"}


def parse_struct(text):
    """-> (relation, head type, tail type). relation is "none" when the model states no relation."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    parts = [clean_phrase(re.sub(r"^\s*(relation|answer)\s*[:：]", "", part, flags=re.I))
             for part in (lines[0] if lines else "").split("|")]
    parts += [""] * (3 - len(parts))
    rel = "none" if parts[0] in NONE_WORDS else parts[0]
    return rel, parts[1] or "entity", parts[2] or "entity"


def typed_text(struct):
    rel, head_type, tail_type = struct
    return f"{head_type} | {rel} | {tail_type}"


# ============================================================================ API model (DeepSeek)
class APIChat:
    """OpenAI-compatible chat client with a thread pool, retries and an append-only cache on disk.

    The cache keeps one line per answered prompt, so an interrupted run resumes and the raw answers
    are kept (rule 5 of docs/collaboration.md)."""

    def __init__(self, api_key, model, base_url, cache_path, workers=32):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)
        self.model, self.workers, self.cache_path = model, workers, cache_path
        self.lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0,
                      "failed": 0}
        self.cache = {}
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self.cache[row["k"]] = row["a"]
                    except Exception:
                        pass                                  # a line cut by an interrupted write
        self._file = open(cache_path, "a", encoding="utf-8")

    def _key(self, system, user, max_tokens):
        return hashlib.sha1(f"{self.model}|{max_tokens}|{system}|{user}".encode("utf-8")).hexdigest()

    def chat(self, user, system=None, max_tokens=60, retries=6):
        key = self._key(system, user, max_tokens)
        if key in self.cache:
            return self.cache[key]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        for attempt in range(retries):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                        max_tokens=max_tokens, temperature=0.0)
                text = r.choices[0].message.content or ""
                u = getattr(r, "usage", None)
                with self.lock:
                    self.usage["calls"] += 1
                    if u is not None:
                        self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                        self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
                        self.usage["cached_prompt_tokens"] += getattr(u, "prompt_cache_hit_tokens", 0) or 0
                    self.cache[key] = text
                    self._file.write(json.dumps({"k": key, "a": text}, ensure_ascii=False) + "\n")
                    if self.usage["calls"] % 200 == 0:
                        self._file.flush()
                return text
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status in (400, 401, 402, 403, 404, 422):      # will not succeed on a retry
                    with self.lock:
                        self.usage["failed"] += 1
                        self.usage.setdefault("errors", {})
                        name = f"{status} {type(e).__name__}"
                        self.usage["errors"][name] = self.usage["errors"].get(name, 0) + 1
                        self.usage.setdefault("last_error", str(e)[:300])
                    return ""
                time.sleep(min(2 ** attempt, 30))
        with self.lock:
            self.usage["failed"] += 1
        return ""

    def missing(self, users, system=None, max_tokens=60):
        """Number of prompts without an answer (failed calls are not cached, so map() retries them)."""
        return sum(self._key(system, u, max_tokens) not in self.cache for u in users)

    def map(self, users, system=None, max_tokens=60, status=None):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        out = [None] * len(users)
        status = status if status is not None else {}
        status.update({"total": len(users), "done": 0})
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(self.chat, u, system, max_tokens): i for i, u in enumerate(users)}
            for f in as_completed(futures):
                out[futures[f]] = f.result()
                status["done"] += 1
        self._file.flush()
        return out


class Background:
    """Runs fn(status) in a thread, so the API calls go on while the GPU works."""

    def __init__(self, fn):
        self.status, self.result, self.error = {"total": 0, "done": 0}, None, None

        def target():
            try:
                self.result = fn(self.status)
            except BaseException as e:               # shown when the result is asked for
                self.error = e
        self.thread = threading.Thread(target=target, daemon=True)
        self.t0 = time.time()
        self.thread.start()

    def wait(self, every=60):
        while self.thread.is_alive():
            self.thread.join(every)
            done, total = self.status.get("done", 0), self.status.get("total", 0)
            rate = done / max(time.time() - self.t0, 1e-6)
            if self.thread.is_alive():
                print(f"  API: {done}/{total} answers, about {(total - done) / max(rate, 1e-6) / 60:.0f} min left",
                      flush=True)
        if self.error is not None:
            raise self.error
        return self.result


# ============================================================================ representations
def fit_normalizer(X_train):
    """Centre on the train mean, then L2-normalise (tames the few huge dimensions of LLM states)."""
    mean = X_train.mean(axis=0, keepdims=True)

    def apply(X):
        X = X - mean
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return apply


def concat(*parts):
    return (np.concatenate(parts, axis=1) / np.sqrt(len(parts))).astype(np.float32)


class TextEncoder:
    def __init__(self, model_name, device):
        from sentence_transformers import SentenceTransformer
        self.model, self.lut = SentenceTransformer(model_name, device=device), {}

    def __call__(self, texts):
        new = sorted({t for t in texts if t not in self.lut})
        if new:
            vectors = self.model.encode(new, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
            self.lut.update(zip(new, np.asarray(vectors, dtype=np.float32)))
        return np.stack([self.lut[t] for t in texts])


# ============================================================================ the repository's M2 and M1
def micro_f1(facts, gold):
    correct = len(gold & set(facts))
    return 2 * correct / (len(facts) + len(gold)) if facts and gold else 0.0


def usable_k(k_values, X):
    """k-means needs clearly more distinct points than clusters (generated phrases repeat a lot)."""
    n_distinct = len(np.unique(np.round(X, 4), axis=0)) if len(X) < 50000 else len(X)
    return [k for k in k_values if k <= n_distinct // 2] or [max(2, min(min(k_values), n_distinct // 2))]


def run_m2(X_train, labels, negative_weight, X_dev, dev_pairs, relations, gold, probe=None):
    """M2 of the repository: detector x one classifier per relation, threshold chosen on dev.

    probe: an already fitted RelationProbe on the same X_train (saves the fit)."""
    from kg_extraction.relations.probe import RelationProbe, best_threshold, facts_above
    probe = probe or RelationProbe(relations).fit(X_train, labels, negative_weight)
    scores = probe.scores(X_dev)
    threshold = best_threshold(dev_pairs, scores, relations, gold)
    facts = facts_above(dev_pairs, scores, relations, threshold)
    return probe, facts, {"threshold": round(threshold, 2), "dev_f1": round(100 * micro_f1(facts, gold), 2)}


def run_m1(probe, X_train, labels, X_dev, dev_pairs, relations, gold, k_values, seed=0):
    """M1 of the repository: same detector, k-means on related train pairs, k and threshold from dev."""
    from kg_extraction.relations.clustering import ClusterTyper
    from kg_extraction.relations.probe import best_threshold, facts_above
    related = np.array([bool(codes) for codes in labels])
    related_labels = [codes for codes in labels if codes]
    detection = probe.detector.predict_proba(X_dev)[:, 1]
    best, tried = None, {}
    for k in usable_k(k_values, X_train[related]):
        typer = ClusterTyper(relations, k, seed).fit(X_train[related], related_labels)
        scores = detection[:, None] * typer.scores(X_dev)
        threshold = best_threshold(dev_pairs, scores, relations, gold)
        facts = facts_above(dev_pairs, scores, relations, threshold)
        f1 = micro_f1(facts, gold)
        tried[k] = round(100 * f1, 2)
        if best is None or f1 > best[0]:
            best = (f1, facts, typer, {"k": k, "threshold": round(threshold, 2)})
    _, facts, typer, settings = best
    settings.update({"dev_f1_by_k": tried, "dev_f1": tried[settings["k"]]})
    return typer, detection, facts, settings


def thresholded(scores, dev_pairs, relations, gold):
    from kg_extraction.relations.probe import best_threshold, facts_above
    threshold = best_threshold(dev_pairs, scores, relations, gold)
    facts = facts_above(dev_pairs, scores, relations, threshold)
    return facts, {"threshold": round(threshold, 2), "dev_f1": round(100 * micro_f1(facts, gold), 2)}


# ============================================================================ naming clusters with an LLM
NAME_SYSTEM = ('You are given pairs of entities that were grouped together automatically because the texts '
               'seem to express the same relation between the head entity and the tail entity. '
               'Choose the one Wikidata property of the list that best describes the relation from head to '
               'tail shared by most pairs. Answer with the property code only (for example P17). '
               'Answer NONE if no property of the list fits.\n\nProperties:\n{relations}')


def name_system(relation_names):
    lines = [f"{code}: {name}" for code, name in sorted(relation_names.items(), key=lambda kv: int(kv[0][1:]))]
    return NAME_SYSTEM.format(relations="\n".join(lines))


def _short(text, n=45):
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


def naming_prompts(clusters, instances, n_clusters, n_examples=8, seed=0):
    """One prompt per cluster, from a random sample of its members. No label is shown."""
    rng = np.random.RandomState(seed)
    clusters = np.asarray(clusters)
    prompts = []
    for c in range(n_clusters):
        members = np.where(clusters == c)[0]
        if len(members) == 0:
            prompts.append(None)
            continue
        picked = rng.choice(members, size=min(n_examples, len(members)), replace=False)
        lines = [f'- head: "{instances[i]["head"]}" ; tail: "{instances[i]["tail"]}" ; text: "{_short(instances[i]["context"])}"'
                 for i in picked]
        prompts.append("Pairs of the group:\n" + "\n".join(lines))
    return prompts


def parse_code(answer, relations):
    for code in re.findall(r"P\d+", answer or ""):
        if code in relations:
            return code
    return None


def name_clusters(chat, clusters, instances, n_clusters, relation_names):
    """-> list with the relation code of each cluster, or None (empty cluster, NONE, or unreadable)."""
    prompts = naming_prompts(clusters, instances, n_clusters)
    system = name_system(relation_names)
    asked = [i for i, p in enumerate(prompts) if p is not None]
    answers = chat.map([prompts[i] for i in asked], system=system, max_tokens=8)
    codes = [None] * n_clusters
    for i, answer in zip(asked, answers):
        codes[i] = parse_code(answer, relation_names)
    return codes


def one_hot_names(codes, relations):
    position = {code: column for column, code in enumerate(relations)}
    matrix = np.zeros((len(codes), len(relations)))
    for cluster, code in enumerate(codes):
        if code is not None:
            matrix[cluster, position[code]] = 1.0
    return matrix


# ============================================================================ run folders
def write_run(directory, card, docs, facts, subset, seed=0, clusters=None):
    """Write a run folder in the format of docs/collaboration.md.

    facts: {(doc index, head, tail, relation): score}; clusters: {(doc index, head, tail): cluster}."""
    os.makedirs(directory, exist_ok=True)
    rows = sorted((str(docs[i]["id"]), h, r, t, round(float(s), 6)) for (i, h, t, r), s in facts.items())
    with open(os.path.join(directory, f"redocred_{subset}_facts_seed{seed}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "h", "r", "t", "score"])
        w.writerows(rows)
    if clusters is not None:
        with open(os.path.join(directory, f"redocred_{subset}_clusters_seed{seed}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "h", "t", "cluster"])
            w.writerows(sorted((str(docs[i]["id"]), h, t, int(c)) for (i, h, t), c in clusters.items()))
    with open(os.path.join(directory, "card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=1)


# ============================================================================ the whole experiment
def subset_docs(docs, ids, limit=None):
    """Documents of a fixed subset of the benchmark, in the order of the split."""
    ids = set(ids)
    chosen = [doc for doc in docs if str(doc["id"]) in ids]
    return chosen[:limit] if limit else chosen


def bert_embeddings(cache_dir, instances, device):
    """The repository's PromptORE embeddings (relations/encoder.py, unchanged), kept on disk."""
    from kg_extraction.relations.encoder import embed_instances
    sig = hashlib.sha1("\x1e".join(inst_key(i) for i in instances).encode("utf-8")).hexdigest()[:10]
    path = os.path.join(cache_dir, f"bert_{sig}.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    embeddings, n_truncated = embed_instances(instances, device=device)
    if n_truncated:
        print(f"  {n_truncated} contexts truncated", flush=True)
    np.save(path + ".tmp.npy", embeddings.astype(np.float16))
    os.replace(path + ".tmp.npy", path)
    return embeddings


def evaluate(repo_dir, run_dir, leaderboard):
    """Score a run folder with the shared evaluator of the repository."""
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "scripts/evaluate.py", os.path.abspath(run_dir),
                           "--leaderboard", os.path.abspath(leaderboard)],
                          cwd=repo_dir, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"evaluate.py failed on {run_dir}:\n{done.stdout}\n{done.stderr}")
    line = [l for l in done.stdout.splitlines() if l.startswith("redocred_")]
    print(f"    {os.path.basename(run_dir):38s} {line[0].split('seeds=1')[-1].strip() if line else ''}", flush=True)


def run_experiment(repo_dir, out_dir, cache_dir, train_docs, dev_docs, allowed, relation_names,
                   llm, llm_name, encoder, encoder_name, chat=None, api_model=None,
                   k_values=(50, 100, 200, 400), label_free_k=100, device="cuda",
                   author="Axel", training_data="train_debug (300 docs)", subset="dev_debug",
                   llm_batch=32, api_max_tokens=40):
    """Run every representation through M2 and M1, write the run folders and score them.

    train_docs / dev_docs: documents of train_debug / dev_debug. allowed: the type filter of the whole
    train split, as in the repository. chat: an APIChat, or None to skip the API methods."""
    from sklearn.cluster import KMeans
    from kg_extraction.evaluation.relations import gold_facts
    from kg_extraction.relations.probe import pair_instances, pair_list, sample_train_pairs

    os.makedirs(out_dir, exist_ok=True)
    cache = Cache(cache_dir)
    leaderboard = os.path.join(out_dir, "leaderboard.csv")
    relations = sorted(relation_names)
    times, settings = {}, {}

    # ---- pairs, exactly as scripts/compare_methods.py and scripts/label_efficiency.py
    all_pairs = pair_list(train_docs, allowed)
    kept, negative_weight = sample_train_pairs(all_pairs, 3, 0)
    train_pairs = [all_pairs[i] for i in kept]
    dev_pairs = pair_list(dev_docs, allowed)
    train_inst, dev_inst = pair_instances(train_docs, train_pairs), pair_instances(dev_docs, dev_pairs)
    labels = [pair[3] for pair in train_pairs]
    related = np.array([bool(codes) for codes in labels])
    gold = gold_facts(dev_docs)
    n = len(train_inst)
    instances = train_inst + dev_inst
    users = [struct_user(inst) for inst in instances]
    print(f"train: {len(train_docs)} documents, {len(all_pairs)} candidate pairs, sample of {n} "
          f"({int(related.sum())} related, weight of an unrelated pair {negative_weight:.2f})\n"
          f"dev:   {len(dev_docs)} documents, {len(dev_pairs)} candidate pairs, {len(gold)} gold facts", flush=True)
    settings["data"] = {"train_docs": len(train_docs), "train_sample": n, "train_related": int(related.sum()),
                        "negative_weight": round(float(negative_weight), 3), "dev_docs": len(dev_docs),
                        "dev_pairs": len(dev_pairs), "dev_gold_facts": len(gold)}

    # ---- the API model answers in the background while the GPU works
    background = None
    if chat is not None:
        background = Background(lambda status: chat.map(users, STRUCT_SYSTEM, api_max_tokens, status))

    # ---- representations
    t0 = time.time()
    print("\n[1] BERT [MASK] embeddings (the repository's PromptORE)", flush=True)
    bert = bert_embeddings(cache_dir, instances, device)
    times["bert"] = time.time() - t0

    t0 = time.time()
    print(f"\n[2] {llm_name}: state and answer of every pair", flush=True)
    states, llm_raw = cache.run(
        "llm", users, lambda batch: llm.answer_with_state(batch, STRUCT_SYSTEM, bs=llm_batch), kind="both")
    times["llm"] = time.time() - t0
    if not np.isfinite(states).all():
        raise ValueError("the LLM states contain NaN or infinite values (try another LLM_NAME)")
    llm_struct = [parse_struct(answer) for answer in llm_raw]

    api_raw = api_struct = None
    if chat is not None:
        t0 = time.time()
        print(f"\n[3] {api_model}: waiting for the answers still missing", flush=True)
        api_raw = background.wait()
        for _ in range(2):                                   # failed calls are tried again
            if chat.missing(users, STRUCT_SYSTEM, api_max_tokens):
                api_raw = chat.map(users, STRUCT_SYSTEM, api_max_tokens)
        missing = chat.missing(users, STRUCT_SYSTEM, api_max_tokens)
        print(f"  usage: {chat.usage}", flush=True)
        if missing > 0.02 * len(users):
            raise RuntimeError(f"{missing} of {len(users)} API calls failed: {chat.usage}. "
                               "Run the cell again: the answers already received are kept.")
        api_struct = [parse_struct(answer) for answer in api_raw]
        times["api_wait_after_gpu"] = time.time() - t0
        settings["api"] = {"missing": missing, **{k: v for k, v in chat.usage.items()}}

    # the answers, readable (rule 5 of docs/collaboration.md: keep what the LLM wrote)
    for split, docs, pairs, offset in (("train", train_docs, train_pairs, 0), ("dev", dev_docs, dev_pairs, n)):
        with open(os.path.join(out_dir, f"llm_answers_{split}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "h", "t", "head", "tail", "gold", "local_llm", "api_llm"])
            for row, (index, head, tail, codes) in enumerate(pairs):
                inst = instances[offset + row]
                w.writerow([docs[index]["id"], head, tail, inst["head"], inst["tail"], " ".join(codes),
                            llm_raw[offset + row].strip().replace("\n", " / "),
                            api_raw[offset + row].strip().replace("\n", " / ") if api_raw else ""])

    def none_share(structs):
        return round(float(np.mean([s[0] == "none" for s in structs])), 3)
    settings["answers"] = {"local_none_share": none_share(llm_struct),
                           "local_distinct_relations": len({s[0] for s in llm_struct})}
    if api_struct:
        settings["answers"].update({"api_none_share": none_share(api_struct),
                                    "api_distinct_relations": len({s[0] for s in api_struct})})
    print(f"\nanswers: {settings['answers']}", flush=True)

    reps = {"bert": (bert[:n], bert[n:], "bert-base-uncased",
                     "PromptORE prompt '{head} [MASK] {tail}.', BERT [MASK] state, the repository's code unchanged")}
    norm = fit_normalizer(states[:n])
    reps["qwen-state"] = (norm(states[:n]), norm(states[n:]), llm_name,
                          "last-layer state of the LLM where its answer starts, centred on train and L2-normalised")
    texts = {"qwen-text": encoder([typed_text(s) for s in llm_struct])}
    reps["qwen-text"] = (texts["qwen-text"][:n], texts["qwen-text"][n:], f"{llm_name}, {encoder_name}",
                         "the LLM writes 'relation | head type | tail type' (relation may be none), "
                         "the answer is embedded by a sentence encoder")
    best_text = "qwen-text"
    if api_struct:
        texts["api-text"] = encoder([typed_text(s) for s in api_struct])
        reps["api-text"] = (texts["api-text"][:n], texts["api-text"][n:], f"{api_model}, {encoder_name}",
                            "an API LLM writes 'relation | head type | tail type' (relation may be none), "
                            "the answer is embedded by a sentence encoder")
        best_text = "api-text"
    bert_norm = fit_normalizer(bert[:n])
    hybrid = concat(bert_norm(bert), texts[best_text])
    reps[f"bert+{best_text}"] = (hybrid[:n], hybrid[n:], f"bert-base-uncased, {reps[best_text][2]}",
                                 f"BERT [MASK] state (centred, L2) concatenated with the {best_text} vector")

    def card(run, method, supervision, models, description, data=training_data):
        return {"run": run, "author": author, "method": method, "supervision": supervision,
                "training_data": data, "models": models, "description": description}

    def save(run, facts, clusters=None, **fields):
        directory = os.path.join(out_dir, run)
        write_run(directory, card(run, **fields), dev_docs, facts, subset, clusters=clusters)
        evaluate(repo_dir, directory, leaderboard)

    protocol = (f"Detector and typing fitted on {training_data}: all related pairs + 3 unrelated per related "
                f"(seed 0, weighted). Threshold chosen on {subset}.")
    related_inst = [inst for inst, codes in zip(train_inst, labels) if codes]

    # ---- M2 and M1 on every representation
    print("\n[4] M2 (classifier) and M1 (clusters) on every representation", flush=True)
    for name, (X_train, X_dev, models, description) in reps.items():
        t0 = time.time()
        print(f"  {name} ({X_train.shape[1]} dimensions)", flush=True)
        probe, facts, m2 = run_m2(X_train, labels, negative_weight, X_dev, dev_pairs, relations, gold)
        save(f"td-{name}-classifier", facts, method=f"{name} + classifier (M2)", supervision="supervised",
             models=models, description=f"{description}. {protocol} Settings: {m2}")

        typer, detection, facts, m1 = run_m1(probe, X_train, labels, X_dev, dev_pairs, relations, gold, k_values)
        assigned = typer.kmeans.predict(X_dev)
        clusters = {(pair[0], pair[1], pair[2]): c for pair, c in zip(dev_pairs, assigned)}
        save(f"td-{name}-clusters", facts, clusters=clusters, method=f"{name} + k-means (M1)",
             supervision="labels_for_mapping", models=models,
             description=f"{description}. {protocol} k chosen on {subset}. Settings: {m1}")
        settings[name] = {"M2": m2, "M1": m1}

        # one relation per cluster: from the train labels (control) or chosen by the API LLM
        k = typer.kmeans.n_clusters
        majority = [relations[column] if typer.precision[c].sum() > 0 else None
                    for c, column in enumerate(typer.precision.argmax(axis=1))]
        _, control = thresholded(detection[:, None] * one_hot_names(majority, relations)[assigned],
                                 dev_pairs, relations, gold)
        settings[name]["M1_one_relation_per_cluster_from_labels"] = control
        if chat is not None:
            codes = name_clusters(chat, typer.kmeans.labels_, related_inst, k, relation_names)
            facts, named = thresholded(detection[:, None] * one_hot_names(codes, relations)[assigned],
                                       dev_pairs, relations, gold)
            named["clusters_named_like_majority_label"] = round(
                float(np.mean([a == b for a, b in zip(codes, majority)])), 3)
            named["clusters_without_name"] = sum(code is None for code in codes)
            save(f"td-{name}-clusters-llmnamed", facts, clusters=clusters,
                 method=f"{name} + k-means, clusters named by an LLM", supervision="labels_for_mapping",
                 models=f"{models}, {api_model}",
                 description=f"{description}. Same detector and clusters as td-{name}-clusters (k={k}), but "
                             f"each cluster gets one relation chosen by {api_model} from 8 of its pairs, without "
                             f"labels. Train labels are only used by the detector. Settings: {named}")
            settings[name]["M1_named_by_llm"] = named
        times[f"fit_{name}"] = time.time() - t0

    # ---- no label at all: the LLM detects (answer is not none), k-means on dev, the API LLM names the clusters
    if chat is not None:
        print("\n[5] Without any label", flush=True)
        for name, structs in (("api-text", api_struct), ("qwen-text", llm_struct)):
            mask = np.array([s[0] != "none" for s in structs[n:]])
            if mask.sum() < 4:
                print(f"  {name}: the LLM finds almost no relation, skipped", flush=True)
                continue
            X = texts[name][n:][mask]
            k = usable_k([label_free_k], X)[0]
            kmeans = KMeans(init="k-means++", n_init=3, n_clusters=k, random_state=0).fit(X)
            picked = [i for i in range(len(dev_pairs)) if mask[i]]
            codes = name_clusters(chat, kmeans.labels_, [dev_inst[i] for i in picked], k, relation_names)
            facts = {(dev_pairs[i][0], dev_pairs[i][1], dev_pairs[i][2], codes[c]): 1.0
                     for i, c in zip(picked, kmeans.labels_) if codes[c] is not None}
            clusters = {(dev_pairs[i][0], dev_pairs[i][1], dev_pairs[i][2]): c for i, c in zip(picked, kmeans.labels_)}
            info = {"k": k, "pairs_called_related": int(mask.sum()),
                    "clusters_without_name": sum(code is None for code in codes),
                    "dev_f1": round(100 * micro_f1(facts, gold), 2)}
            save(f"td-{name}-labelfree", facts, clusters=clusters, data="none",
                 method=f"{name}: LLM detection + k-means + clusters named by an LLM", supervision="none",
                 models=f"{reps[name][2]}, {api_model}",
                 description=f"No label. A pair is related when the LLM answer is not none. k-means (k={k}, fixed) "
                             f"on the answers of the {subset} pairs called related, each cluster named by "
                             f"{api_model} with one of the 96 relations. Settings: {info}")
            settings[f"{name}-labelfree"] = info

    settings["seconds"] = {k: round(v) for k, v in times.items()}
    with open(os.path.join(out_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)
    return settings


# ============================================================================ the larger run (train 1000 documents, full dev)
def embed_cached(cache, name, instances, device):
    """The repository's PromptORE embeddings, in resumable chunks of 20,000 pairs."""
    from kg_extraction.relations.encoder import embed_instances

    def fn(batch):
        embeddings, _ = embed_instances([key_inst(key) for key in batch], device=device)
        return embeddings
    return cache.run(name, [inst_key(i) for i in instances], fn, chunk=20000)


def api_answers(chat, users, max_tokens, what, max_new_calls=None):
    """Answers of the API model for every prompt, retrying the failed calls; refuses to start a run that
    would make more new calls than the budget allows."""
    new = chat.missing(users, STRUCT_SYSTEM, max_tokens)
    print(f"  {what}: {len(users)} prompts, {len(users) - new} already answered, {new} new API calls "
          f"(about {new * 215 / 1e6:.1f} million input tokens)", flush=True)
    if max_new_calls is not None and new > max_new_calls:
        raise RuntimeError(f"{new} new API calls are above the budget of {max_new_calls} (MAX_NEW_CALLS).")
    background = Background(lambda status: chat.map(users, STRUCT_SYSTEM, max_tokens, status))
    return background, new


def finish_api(chat, background, users, max_tokens):
    answers = background.wait()
    for _ in range(2):                                           # failed calls are tried again
        if chat.missing(users, STRUCT_SYSTEM, max_tokens):
            answers = chat.map(users, STRUCT_SYSTEM, max_tokens)
    missing = chat.missing(users, STRUCT_SYSTEM, max_tokens)
    if missing > 0.01 * len(users):
        raise RuntimeError(f"{missing} of {len(users)} API calls failed: {chat.usage}. "
                           "Run the cell again: the answers already received are kept.")
    return answers


def run_final(repo_dir, out_dir, cache_dir, train_a, train_b, dev_docs, tune_ids, allowed, relation_names,
              encoder, encoder_name, chat, api_model, k_values=(100, 200, 400), device="cuda",
              author="Axel", prefix="t1k", prefilter=True, thresholds=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
              max_loss=0.5, max_new_calls=230000, training_data=None, api_max_tokens=40):
    """BERT and BERT+API-text through M2 and M1, trained on train_a + train_b, scored on dev_docs.

    train_a: documents whose pairs were already answered by an earlier run (the sample is drawn separately
    on train_a and on train_b with the repository's rule, all related pairs + 3 unrelated per related,
    seed 0, and the unrelated pairs are weighted by the share they stand for).
    tune_ids: ids of the dev documents used to choose the pair filter (the dev_debug subset).
    prefilter: only the dev pairs whose BERT detection probability reaches a threshold are sent to the
    API, and the threshold is the largest that costs at most max_loss F1 points on the tune documents."""
    from kg_extraction.evaluation.relations import gold_facts
    from kg_extraction.relations.probe import RelationProbe, pair_instances, pair_list, sample_train_pairs

    os.makedirs(out_dir, exist_ok=True)
    cache = Cache(cache_dir)
    leaderboard = os.path.join(out_dir, "leaderboard.csv")
    relations = sorted(relation_names)
    train_docs = list(train_a) + list(train_b)
    settings, times = {}, {}
    training_data = training_data or f"{len(train_docs)} train documents"

    # ---- pairs and train sample
    pairs_a = pair_list(train_a, allowed)
    pairs_b = [(i + len(train_a), h, t, c) for i, h, t, c in pair_list(train_b, allowed)] if train_b else []
    kept_a, _ = sample_train_pairs(pairs_a, 3, 0)
    kept_b = sample_train_pairs(pairs_b, 3, 0)[0] if pairs_b else []
    train_pairs = [pairs_a[i] for i in kept_a] + [pairs_b[i] for i in kept_b]
    labels = [pair[3] for pair in train_pairs]
    related = np.array([bool(codes) for codes in labels])
    n_unrelated = sum(1 for p in pairs_a + pairs_b if not p[3])
    negative_weight = n_unrelated / max(int((~related).sum()), 1)
    train_inst = pair_instances(train_docs, train_pairs)
    dev_pairs = pair_list(dev_docs, allowed)
    dev_inst = pair_instances(dev_docs, dev_pairs)
    gold = gold_facts(dev_docs)
    tune_index = {i for i, doc in enumerate(dev_docs) if str(doc["id"]) in set(tune_ids)}
    tune_rows = [r for r, pair in enumerate(dev_pairs) if pair[0] in tune_index]
    gold_tune = {f for f in gold if f[0] in tune_index}
    n = len(train_inst)
    print(f"train: {len(train_docs)} documents, sample of {n} pairs ({int(related.sum())} related, weight of an "
          f"unrelated pair {negative_weight:.2f})\n"
          f"dev:   {len(dev_docs)} documents, {len(dev_pairs)} candidate pairs, {len(gold)} gold facts; "
          f"{len(tune_index)} documents ({len(tune_rows)} pairs) used to choose the pair filter", flush=True)
    settings["data"] = {"train_docs": len(train_docs), "train_sample": n, "train_related": int(related.sum()),
                        "negative_weight": round(float(negative_weight), 3), "dev_docs": len(dev_docs),
                        "dev_pairs": len(dev_pairs), "dev_gold_facts": len(gold)}
    users_train = [struct_user(i) for i in train_inst]

    # ---- the API answers the train sample while the GPU embeds
    print("\n[1] train sample: API answers (background) and BERT [MASK] embeddings", flush=True)
    api_train_job, _ = api_answers(chat, users_train, api_max_tokens, "train sample", max_new_calls)
    t0 = time.time()
    bert_train = embed_cached(cache, "bert_train", train_inst, device)
    bert_dev = embed_cached(cache, "bert_dev", dev_inst, device)
    times["bert"] = time.time() - t0
    t0 = time.time()
    api_train = finish_api(chat, api_train_job, users_train, api_max_tokens)
    times["api_train_wait"] = time.time() - t0
    struct_train = [parse_struct(a) for a in api_train]
    text_train = encoder([typed_text(s) for s in struct_train])
    bert_norm = fit_normalizer(bert_train)
    hybrid_train = concat(bert_norm(bert_train), text_train)
    print(f"  usage so far: {chat.usage}", flush=True)

    # ---- the BERT detector: the baseline, and the filter of the dev pairs
    print("\n[2] BERT baseline fitted; pair filter chosen on the tune documents", flush=True)
    probe_bert = RelationProbe(relations).fit(bert_train, labels, negative_weight)
    detection_dev = probe_bert.detector.predict_proba(bert_dev)[:, 1]
    probe_hybrid = RelationProbe(relations).fit(hybrid_train, labels, negative_weight)
    users_dev_all = [struct_user(i) for i in dev_inst]
    users_tune = [users_dev_all[r] for r in tune_rows]
    job, _ = api_answers(chat, users_tune, api_max_tokens, "tune documents", max_new_calls)
    api_tune = finish_api(chat, job, users_tune, api_max_tokens)
    text_tune = encoder([typed_text(parse_struct(a)) for a in api_tune])
    x_tune = concat(bert_norm(bert_dev[tune_rows]), text_tune)
    scores_tune = probe_hybrid.scores(x_tune)
    pairs_tune = [dev_pairs[r] for r in tune_rows]
    table, threshold_filter = [], 0.0
    base, stopped = None, False
    for t in (0.0,) + tuple(thresholds):
        mask = detection_dev[tune_rows] >= t
        _, info = thresholded(scores_tune * mask[:, None], pairs_tune, relations, gold_tune)
        base = info["dev_f1"] if base is None else base
        share = float((detection_dev >= t).mean())
        table.append({"filter_threshold": t, "share_of_dev_pairs_kept": round(share, 3), "tune_f1": info["dev_f1"]})
        print(f"  filter {t:<6}: keeps {100 * share:5.1f}% of the dev pairs, F1 on the tune documents {info['dev_f1']}",
              flush=True)
        if prefilter and not stopped:
            if base - info["dev_f1"] <= max_loss:
                threshold_filter = t
            else:
                stopped = True                          # the next thresholds keep even fewer pairs
    settings["pair_filter"] = {"used": bool(prefilter), "chosen_threshold": threshold_filter, "table": table,
                               "max_loss_f1_points": max_loss}
    print(f"  chosen: {threshold_filter if prefilter else 'no filter'}", flush=True)

    # ---- the API answers the kept dev pairs
    keep = detection_dev >= threshold_filter
    keep[tune_rows] = True
    kept_rows = np.nonzero(keep)[0]
    print(f"\n[3] dev pairs: {len(kept_rows)} of {len(dev_pairs)} go to the API", flush=True)
    users_kept = [users_dev_all[r] for r in kept_rows]
    job, new_calls = api_answers(chat, users_kept, api_max_tokens, "dev pairs", max_new_calls)
    api_kept = finish_api(chat, job, users_kept, api_max_tokens)
    print(f"  usage: {chat.usage}", flush=True)
    settings["api"] = {"dev_pairs_sent": int(len(kept_rows)), **chat.usage}
    struct_kept = [parse_struct(a) for a in api_kept]
    text_kept = encoder([typed_text(s) for s in struct_kept])
    hybrid_kept = concat(bert_norm(bert_dev[kept_rows]), text_kept)
    kept_pairs = [dev_pairs[r] for r in kept_rows]

    # ---- the answers, readable
    for split, docs, pairs, insts, answers in (("train", train_docs, train_pairs, train_inst, api_train),
                                                ("dev", dev_docs, kept_pairs, [dev_inst[r] for r in kept_rows], api_kept)):
        with open(os.path.join(out_dir, f"llm_answers_{split}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "h", "t", "head", "tail", "gold", "api_llm"])
            for (index, head, tail, codes), inst, answer in zip(pairs, insts, answers):
                w.writerow([docs[index]["id"], head, tail, inst["head"], inst["tail"], " ".join(codes),
                            answer.strip().replace("\n", " / ")])
    settings["answers"] = {"api_none_share_train": round(float(np.mean([s[0] == "none" for s in struct_train])), 3),
                           "api_none_share_dev": round(float(np.mean([s[0] == "none" for s in struct_kept])), 3),
                           "api_distinct_relations": len({s[0] for s in struct_kept + struct_train})}

    # ---- M2 and M1
    def save(run, facts, clusters=None, **fields):
        directory = os.path.join(out_dir, run)
        card = {"run": run, "author": author, "training_data": training_data, **fields}
        write_run(directory, card, dev_docs, facts, "dev", clusters=clusters)
        evaluate(repo_dir, directory, leaderboard)

    protocol = (f"Detector and typing fitted on {training_data}: all related pairs + 3 unrelated per related "
                f"(seed 0, weighted). Fact threshold chosen on dev.")
    filtered = (f" Only the dev pairs whose BERT detection probability is at least {threshold_filter} "
                f"({100 * len(kept_rows) / len(dev_pairs):.0f}% of them) were given to the API; the others have no fact."
                if threshold_filter > 0 else "")
    reps = {"bert": (bert_train, bert_dev, np.arange(len(dev_pairs)), "bert-base-uncased", probe_bert,
                     "PromptORE prompt '{head} [MASK] {tail}.', BERT [MASK] state, the repository's code unchanged"),
            "bert+api-text": (hybrid_train, hybrid_kept, kept_rows, f"bert-base-uncased, {api_model}, {encoder_name}",
                              probe_hybrid, "BERT [MASK] state (centred, L2) concatenated with the API answer vector."
                              + filtered)}
    print("\n[4] M2 (classifier) and M1 (clusters)", flush=True)
    for name, (X_train, X_dev, rows, models, probe, description) in reps.items():
        t0 = time.time()
        pairs = [dev_pairs[r] for r in rows]
        print(f"  {name} ({X_train.shape[1]} dimensions, {len(pairs)} dev pairs)", flush=True)
        probe, facts, m2 = run_m2(X_train, labels, negative_weight, X_dev, pairs, relations, gold, probe=probe)
        save(f"{prefix}-{name}-classifier", facts, method=f"{name} + classifier (M2)", supervision="supervised",
             models=models, description=f"{description} {protocol} Settings: {m2}")
        typer, _, facts, m1 = run_m1(probe, X_train, labels, X_dev, pairs, relations, gold, list(k_values))
        assigned = typer.kmeans.predict(X_dev)
        clusters = {(p[0], p[1], p[2]): c for p, c in zip(pairs, assigned)}
        save(f"{prefix}-{name}-clusters", facts, clusters=clusters, method=f"{name} + k-means (M1)",
             supervision="labels_for_mapping", models=models,
             description=f"{description} {protocol} k chosen on dev. Settings: {m1}")
        settings[name] = {"M2": m2, "M1": m1}
        times[f"fit_{name}"] = time.time() - t0

    settings["seconds"] = {k: round(v) for k, v in times.items()}
    with open(os.path.join(out_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)
    return settings
