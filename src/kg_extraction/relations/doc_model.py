"""Document-level relation model over known entities, on frozen token features.

The base is the ATLOP head of pair_classifier.py, computed document by document
so that trainable parts can sit between the frozen encoder and the head:

- token layers: Transformer layers over the frozen token states, which adapt the
  states to relations without touching the encoder;
- pair features: the entity types, the number of mentions and the sentence
  distance of the pair, as learned embeddings;
- pair layers: reasoning over the entity-pair grid of the document. The vector of
  pair (i, j) is updated from the pairs (i, k) and (k, j) and from the pair (j, i),
  so the model can learn rules such as "located in" + "country" -> "country".

With every option off, the model is the same as PairClassifier.

Partial fine-tuning: with top_layers, the last layers of the encoder are trained too.
The documents then carry the states that enter these layers (doc_encoder.encode_lower)
instead of the final states.
"""
import torch
from torch import nn
from torch.nn import functional as F

from kg_extraction.relations.pair_classifier import document_pairs, pair_labels

DISTANCE_BOUNDS = (1, 2, 3, 4, 6, 9)
MAX_MENTIONS = 5


def sentence_distance(head, tail):
    """Return the smallest sentence distance between a mention of each entity."""
    return min(abs(first["sent_id"] - second["sent_id"])
               for first in head["mentions"] for second in tail["mentions"])


def prepare_document(doc, features, entity_types, relations=None):
    """Return the tensors of one document, on CPU.

    features is the output of doc_encoder.encode_document. Labels are added when
    relations (the sorted relation codes) is given.
    """
    entities = doc["entities"]
    positions = features["positions"]
    width = max(len(entity_positions) for entity_positions in positions)
    mentions = torch.zeros(len(positions), width, dtype=torch.long)
    mask = torch.zeros(len(positions), width, dtype=torch.bool)
    for index, entity_positions in enumerate(positions):
        mentions[index, :len(entity_positions)] = torch.tensor(entity_positions)
        mask[index, :len(entity_positions)] = True
    if "lower" in features:
        encoded = {"lower": features["lower"], "windows": features["windows"],
                   "length": features["length"]}
    else:
        attention = features["attention"].float()
        entity_attention, row = [], 0
        for entity_positions in positions:
            entity_attention.append(attention[row:row + len(entity_positions)].mean(0))
            row += len(entity_positions)
        encoded = {"states": features["states"],
                   "entity_attention": torch.stack(entity_attention).half()}
    pairs = document_pairs(doc)
    distances = torch.tensor([sentence_distance(entities[head], entities[tail])
                              for head, tail in pairs.tolist()])
    item = {
        **encoded,
        "mentions": mentions, "mention_mask": mask, "pairs": pairs,
        "types": torch.tensor([entity_types.index(entity["type"]) for entity in entities]),
        "counts": torch.tensor([min(len(entity["mentions"]), MAX_MENTIONS) - 1
                                for entity in entities]),
        "distances": torch.bucketize(distances, torch.tensor(DISTANCE_BOUNDS), right=True),
    }
    if relations is not None:
        item["labels"] = pair_labels(doc, pairs, relations).to(torch.uint8)
    return item


class PairLayer(nn.Module):
    """One reasoning step over the (entities, entities, width) grid of pair vectors."""

    def __init__(self, width, inner=64, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.left = nn.Linear(width, inner)
        self.right = nn.Linear(width, inner)
        self.merge = nn.Linear(3 * inner + width, width)
        self.feed_forward = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                          nn.GELU(), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, grid, mask):
        normed = self.norm(grid)
        left = self.left(normed) * mask
        right = self.right(normed) * mask
        size = grid.shape[0]
        chain = torch.einsum("ikc,kjc->ijc", left, right) / size
        same_tail = torch.einsum("ikc,jkc->ijc", left, right) / size
        same_head = torch.einsum("kic,kjc->ijc", left, right) / size
        update = self.merge(torch.cat([chain, same_tail, same_head, normed.transpose(0, 1)], -1))
        grid = grid + self.dropout(update) * mask
        return grid + self.dropout(self.feed_forward(grid)) * mask


class TunedEncoder(nn.Module):
    """The last layers of a BERT-like encoder, trained with the relation model.

    It returns what the frozen cache holds otherwise: the token states of the document
    and, for each entity, the attention of its mention markers over the tokens (last
    layer, mean over heads, each marker row normalized). Windows are averaged.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, item):
        length = item["length"]
        device = item["mentions"].device
        states = attention = None
        counts = torch.zeros(length, device=device)
        for (start, end), lower in zip(item["windows"], item["lower"]):
            hidden = lower.float()[None]
            for layer in self.layers:
                attended, weights = layer.attention(hidden, output_attentions=True)
                hidden = layer.output(layer.intermediate(attended), attended)
            if states is None:
                states = hidden.new_zeros(length, hidden.shape[-1])
                attention = hidden.new_zeros(length, length)
            states = states.index_add(0, torch.arange(start, end, device=device), hidden[0, 1:-1])
            padded = hidden.new_zeros(length, length)
            padded[start:end, start:end] = weights[0].mean(0)[1:-1, 1:-1]
            attention = attention + padded
            counts[start:end] += 1
        states = states / counts[:, None]
        rows = attention[item["mentions"]]
        rows = rows / rows.sum(-1, keepdim=True).clamp_min(1e-12)
        mask = item["mention_mask"][..., None]
        return states, (rows * mask).sum(1) / mask.sum(1)


class DocRelationModel(nn.Module):
    def __init__(self, n_relations, hidden=768, n_types=6, token_layers=0, pair_layers=0,
                 pair_features=False, embedding=768, block=64, pair_width=256, dropout=0.1,
                 top_layers=None):
        """top_layers: the encoder layers to train (doc_encoder.pretrained_top_layers)."""
        super().__init__()
        self.settings = dict(n_relations=n_relations, hidden=hidden, n_types=n_types,
                             token_layers=token_layers, pair_layers=pair_layers,
                             pair_features=pair_features, embedding=embedding, block=block,
                             pair_width=pair_width, dropout=dropout)
        self.encoder = TunedEncoder(top_layers) if top_layers is not None else None
        self.dropout = nn.Dropout(dropout)
        self.block = block
        self.tokens = None
        if token_layers:
            layer = nn.TransformerEncoderLayer(hidden, 8, 2 * hidden, dropout, batch_first=True,
                                               norm_first=True)
            self.tokens = nn.TransformerEncoder(layer, token_layers, enable_nested_tensor=False)
        extra = 0
        self.pair_features = pair_features
        if pair_features:
            self.types = nn.Embedding(n_types, 32)
            self.counts = nn.Embedding(MAX_MENTIONS, 16)
            self.distances = nn.Embedding(len(DISTANCE_BOUNDS) + 1, 32)
            extra = 32 + 16 + 32
        self.head = nn.Linear(2 * hidden + extra, embedding)
        self.tail = nn.Linear(2 * hidden + extra, embedding)
        self.bilinear = nn.Linear(embedding * block, n_relations + 1)
        self.pairs = None
        if pair_layers:
            self.pair_input = nn.Linear(2 * embedding + n_relations + 1, pair_width)
            self.pairs = nn.ModuleList(PairLayer(pair_width, dropout=dropout)
                                       for _ in range(pair_layers))
            self.pair_output = nn.Sequential(nn.LayerNorm(pair_width),
                                             nn.Linear(pair_width, n_relations + 1))
            nn.init.zeros_(self.pair_output[1].weight)
            nn.init.zeros_(self.pair_output[1].bias)

    def forward(self, item):
        """Return the (pairs, relations + 1) logits of one prepared document."""
        if self.encoder is not None:
            states, attention = self.encoder(item)
        else:
            states, attention = item["states"].float(), item["entity_attention"].float()
        states = self.dropout(states)
        if self.tokens is not None:
            states = self.tokens(states[None])[0]
        mentions = states[item["mentions"]].masked_fill(~item["mention_mask"][..., None],
                                                        float("-inf"))
        entities = torch.logsumexp(mentions, 1)
        heads, tails = item["pairs"][:, 0], item["pairs"][:, 1]
        weights = attention[heads] * attention[tails]
        contexts = (weights / weights.sum(1, keepdim=True).clamp_min(1e-12)) @ states
        head_inputs = [entities[heads], contexts]
        tail_inputs = [entities[tails], contexts]
        if self.pair_features:
            distances = self.distances(item["distances"])
            head_inputs += [self.types(item["types"])[heads], self.counts(item["counts"])[heads],
                            distances]
            tail_inputs += [self.types(item["types"])[tails], self.counts(item["counts"])[tails],
                            distances]
        head = torch.tanh(self.head(torch.cat(head_inputs, 1)))
        tail = torch.tanh(self.tail(torch.cat(tail_inputs, 1)))
        grouped = (head.view(len(head), -1, self.block).unsqueeze(3)
                   * tail.view(len(tail), -1, self.block).unsqueeze(2)).view(len(head), -1)
        logits = self.bilinear(grouped)
        if self.pairs is None:
            return logits
        size = len(entities)
        vectors = self.pair_input(torch.cat([head, tail, logits], 1))
        grid = vectors.new_zeros(size, size, vectors.shape[1])
        grid[heads, tails] = vectors
        mask = (1 - torch.eye(size, device=grid.device))[..., None]
        for layer in self.pairs:
            grid = layer(grid, mask)
        return logits + self.pair_output(grid[heads, tails])


def adaptive_focal_loss(logits, labels, gamma=1.0):
    """Adaptive focal loss (Tan et al., 2022), for the long tail of relations.

    Each gold relation is compared with the threshold class alone, and its loss is
    weighted by (1 - p) ** gamma, so that relations the model already ranks above
    the threshold weigh less than the hard, often rare, ones. The part that keeps
    the threshold above the other relations is the one of the ATLOP loss.
    """
    labels = labels.clone()
    labels[:, 0] = 0
    log_p = F.logsigmoid(logits - logits[:, :1])
    loss_above = -((1 - log_p.exp()) ** gamma * log_p * labels).sum(1)
    loss_below = -torch.log_softmax(logits - labels * 1e30, -1)[:, 0]
    return (loss_above + loss_below).mean()


def to_device(item, device):
    """Move the tensors of a prepared document, and its lists of tensors, to the device."""
    moved = {}
    for name, value in item.items():
        if torch.is_tensor(value):
            value = value.to(device)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            value = [tensor.to(device) for tensor in value]
        moved[name] = value
    return moved


@torch.no_grad()
def predict_scores(model, items, docs, relations, min_score=0.02):
    """Return {(document index, head id, tail id, relation code): score} above min_score.

    The score is sigmoid(logit - threshold logit): 0.5 is the learned threshold.
    """
    model.eval()
    device = next(model.parameters()).device
    facts = {}
    for index, item in enumerate(items):
        logits = model(to_device(item, device))
        scores = torch.sigmoid(logits[:, 1:] - logits[:, :1]).cpu()
        ids = [entity["id"] for entity in docs[index]["entities"]]
        pairs = item["pairs"].tolist()
        for row, column in torch.nonzero(scores >= min_score).tolist():
            head, tail = pairs[row]
            facts[(index, ids[head], ids[tail], relations[column])] = float(scores[row, column])
    return facts


def best_threshold(facts, gold):
    """Return the score threshold with the best F1 on scored facts, and that F1 in percent."""
    ranked = sorted(facts.items(), key=lambda fact: -fact[1])
    best, correct = (0.5, 0.0), 0
    for rank, (fact, score) in enumerate(ranked, start=1):
        correct += fact in gold
        f1 = 2 * correct / (rank + len(gold))
        if f1 > best[1]:
            best = (score, f1)
    return best[0], round(100 * best[1], 2)
