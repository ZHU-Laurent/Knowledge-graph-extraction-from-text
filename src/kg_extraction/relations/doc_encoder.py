"""Encode whole documents once with a frozen encoder, for relation extraction with known entities.

Each mention is wrapped in "*" markers, as in ATLOP, and is represented by the
token of its opening marker. A document longer than the encoder window is split
into overlapping windows, and the states and attention of the windows are averaged.

For partial fine-tuning, encode_lower stops before the last layers: it caches the
states that enter them, and doc_model.TunedEncoder runs these layers in training.
"""
import copy
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-cased"
MARKER = "*"
WINDOW = 512
STRIDE = 256
LAST_LAYERS = 4


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def word_tokenizer(tokenizer):
    """Return a function that splits one word into pieces, as if a space came before it.

    Byte-level BPE tokenizers (RoBERTa) mark the space inside the piece, so the word
    needs its leading space to get the pieces it has in running text.
    """
    if tokenizer.tokenize(" a") != tokenizer.tokenize("a"):
        return lambda word: tokenizer.tokenize(" " + word)
    return tokenizer.tokenize


def tokenize_document(tokenizer, doc):
    """Return the word piece ids of a document and the marker position of each mention.

    Positions are grouped by entity, in the order of doc["entities"].
    """
    tokenize = word_tokenizer(tokenizer)
    marker = tokenize(MARKER)
    starts, ends = {}, set()
    for entity_index, entity in enumerate(doc["entities"]):
        for mention in entity["mentions"]:
            start, end = mention["pos"]
            starts.setdefault((mention["sent_id"], start), []).append(entity_index)
            ends.add((mention["sent_id"], end - 1))

    pieces = []
    positions = [[] for _ in doc["entities"]]
    for sent_id, sentence in enumerate(doc["sents"]):
        for word_id, word in enumerate(sentence):
            word_pieces = tokenize(word) or [tokenizer.unk_token]
            if (sent_id, word_id) in starts:
                for entity_index in starts[(sent_id, word_id)]:
                    positions[entity_index].append(len(pieces))
                word_pieces = [*marker, *word_pieces]
            if (sent_id, word_id) in ends:
                word_pieces = [*word_pieces, *marker]
            pieces.extend(word_pieces)
    return tokenizer.convert_tokens_to_ids(pieces), positions


def windows(length, window=WINDOW - 2, stride=STRIDE):
    """Return the (start, end) token ranges that cover a sequence of a given length."""
    if length <= window:
        return [(0, length)]
    ranges = list(range(0, length - window, stride)) + [length - window]
    return [(start, start + window) for start in ranges]


@torch.no_grad()
def encode_document(model, tokenizer, doc, device):
    """Return the token states and the attention rows of the mention markers of a document.

    states is (tokens, hidden), the mean of the last LAST_LAYERS layers. attention is
    (mentions, tokens), from the last layer averaged over heads, one row per marker
    in the order of the flattened positions.
    """
    ids, positions = tokenize_document(tokenizer, doc)
    markers = [position for entity in positions for position in entity]
    length = len(ids)
    states = torch.zeros(length, model.config.hidden_size, device=device)
    attention = torch.zeros(length, length, device=device)
    counts = torch.zeros(length, device=device)
    for start, end in windows(length):
        window_ids = [tokenizer.cls_token_id, *ids[start:end], tokenizer.sep_token_id]
        output = model(torch.tensor([window_ids], device=device),
                       output_hidden_states=True, output_attentions=True)
        layers = torch.stack(output.hidden_states[-LAST_LAYERS:]).mean(0)[0]
        states[start:end] += layers[1:-1]
        attention[start:end, start:end] += output.attentions[-1][0].mean(0)[1:-1, 1:-1]
        counts[start:end] += 1
    states /= counts[:, None]
    attention = attention[markers]
    attention /= attention.sum(1, keepdim=True).clamp_min(1e-12)
    return {"states": states.half().cpu(), "attention": attention.half().cpu(),
            "positions": positions}


@torch.no_grad()
def encode_lower(model, tokenizer, doc, device, top_layers):
    """Return the states that enter the last top_layers layers, one tensor per window.

    Each window keeps its [CLS] and [SEP] states: lower[i] is (end - start + 2, hidden)
    for windows[i] = (start, end).
    """
    ids, positions = tokenize_document(tokenizer, doc)
    ranges, lower = windows(len(ids)), []
    for start, end in ranges:
        window_ids = [tokenizer.cls_token_id, *ids[start:end], tokenizer.sep_token_id]
        output = model(torch.tensor([window_ids], device=device), output_hidden_states=True)
        lower.append(output.hidden_states[-(top_layers + 1)][0].half().cpu())
    return {"lower": lower, "windows": ranges, "length": len(ids), "positions": positions}


def pretrained_top_layers(model_name, top_layers):
    """Return a copy of the last layers of a BERT-like encoder, to be trained."""
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    return copy.deepcopy(model.encoder.layer[-top_layers:])


def encode_documents(docs, model_name=MODEL_NAME, device=None, top_layers=0):
    device = device or default_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager").to(device).eval()
    if top_layers:
        return [encode_lower(model, tokenizer, doc, device, top_layers)
                for doc in tqdm(docs, desc="Encoding", mininterval=30)]
    return [encode_document(model, tokenizer, doc, device)
            for doc in tqdm(docs, desc="Encoding", mininterval=30)]


def cached_features(path, docs, model_name=MODEL_NAME, top_layers=0):
    """Return the features of documents, encoding them only if the cache is missing or stale.

    With top_layers, the features are those of encode_lower.
    """
    path = Path(path)
    titles = [doc["title"] for doc in docs]
    if path.exists():
        cached = torch.load(path, weights_only=False)
        if (cached["model"] == model_name and cached["titles"] == titles
                and cached.get("top_layers", 0) == top_layers):
            return cached["features"]
    features = encode_documents(docs, model_name, top_layers=top_layers)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_name, "titles": titles, "top_layers": top_layers,
                "features": features}, path)
    return features
