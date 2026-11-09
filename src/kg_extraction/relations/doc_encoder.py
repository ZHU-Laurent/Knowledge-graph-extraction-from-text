"""Encode whole documents once with a frozen BERT, for relation extraction with known entities.

Each mention is wrapped in "*" markers, as in ATLOP, and is represented by the
token of its opening marker. A document longer than the encoder window is split
into overlapping windows, and the states and attention of the windows are averaged.
"""
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


def tokenize_document(tokenizer, doc):
    """Return the word piece ids of a document and the marker position of each mention.

    Positions are grouped by entity, in the order of doc["entities"].
    """
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
            word_pieces = tokenizer.tokenize(word) or [tokenizer.unk_token]
            if (sent_id, word_id) in starts:
                for entity_index in starts[(sent_id, word_id)]:
                    positions[entity_index].append(len(pieces))
                word_pieces = [MARKER, *word_pieces]
            if (sent_id, word_id) in ends:
                word_pieces = [*word_pieces, MARKER]
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


def encode_documents(docs, device=None):
    device = device or default_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device).eval()
    return [encode_document(model, tokenizer, doc, device)
            for doc in tqdm(docs, desc="Encoding", mininterval=30)]


def cached_features(path, docs):
    """Return the features of documents, encoding them only if the cache is missing or stale."""
    path = Path(path)
    titles = [doc["title"] for doc in docs]
    if path.exists():
        cached = torch.load(path, weights_only=False)
        if cached["model"] == MODEL_NAME and cached["titles"] == titles:
            return cached["features"]
    features = encode_documents(docs)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": MODEL_NAME, "titles": titles, "features": features}, path)
    return features
