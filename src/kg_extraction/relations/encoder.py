"""Compute PromptORE relation embeddings with an untrained BERT."""
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizerFast

MODEL_NAME = "bert-base-uncased"
PROMPT_WITH_PERIOD = "{head} [MASK] {tail}."
PROMPT_WITHOUT_PERIOD = "{head} [MASK] {tail}"


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_instance(tokenizer, context, head, tail, template, max_length):
    """Return token ids and whether the context was truncated.

    Only the context is truncated, so the prompt and its [MASK] always remain.
    """
    prompt_ids = tokenizer.encode(template.format(head=head, tail=tail), add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    room = max_length - len(prompt_ids) - 2
    if room < 0:
        raise ValueError(f"Prompt longer than max_length: {head!r}, {tail!r}")
    truncated = len(context_ids) > room
    ids = [tokenizer.cls_token_id, *context_ids[:room], *prompt_ids, tokenizer.sep_token_id]
    return ids, truncated


def embed_instances(instances, template=PROMPT_WITH_PERIOD, max_length=512,
                    batch_size=32, device=None):
    """Return the [MASK] embeddings of the instances and the truncated count.

    Each instance is a dict with the keys context, head and tail.
    """
    device = device or default_device()
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).to(device).eval()

    encoded = [
        encode_instance(tokenizer, item["context"], item["head"], item["tail"],
                        template, max_length)
        for item in tqdm(instances, desc="Tokenizing")
    ]
    n_truncated = sum(truncated for _, truncated in encoded)

    # Sorting by length keeps batches short, and the order is restored at the end.
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index][0]))
    embeddings = np.zeros((len(encoded), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(order), batch_size), desc="Embedding"):
            indices = order[start:start + batch_size]
            batch = tokenizer.pad([{"input_ids": encoded[index][0]} for index in indices],
                                  return_tensors="pt").to(device)
            output = model(**batch).last_hidden_state
            rows, columns = (batch["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(rows) != len(indices):
                raise ValueError("Each prompt must contain exactly one [MASK]")
            embeddings[indices] = output[rows, columns].cpu().numpy()
    return embeddings, n_truncated
