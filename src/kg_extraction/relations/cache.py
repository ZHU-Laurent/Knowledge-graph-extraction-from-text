"""Cache PromptORE embeddings on disk, in float16."""
import json
from pathlib import Path

import numpy as np

from kg_extraction.relations.encoder import embed_instances


def cached_embeddings(path, instances):
    """Return the embeddings of instances, computing them only if the cache is missing.

    The cache stores the instances next to the embeddings, and is recomputed when
    they differ, for example after a new S1 or S2 output.
    """
    path = Path(path)
    keys_path = path.with_suffix(".json")
    if path.exists() and keys_path.exists():
        if json.loads(keys_path.read_text(encoding="utf-8")) == instances:
            return np.load(path).astype(np.float32)
    embeddings, n_truncated = embed_instances(instances)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float16))
    keys_path.write_text(json.dumps(instances), encoding="utf-8")
    if n_truncated:
        print(f"{n_truncated} contexts truncated", flush=True)
    return embeddings
