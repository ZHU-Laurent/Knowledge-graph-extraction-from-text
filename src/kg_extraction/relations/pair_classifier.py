"""Relation classifier over known entities, on frozen document features.

The head follows ATLOP (Zhou et al., 2021): entities are pooled from their mentions
with logsumexp, each pair gets a context vector from the attention that both
entities pay to the document, and a grouped bilinear layer scores every relation.
Class 0 is a learned threshold: a relation holds when its logit passes the logit
of class 0, which handles "no relation" and pairs with several relations.

The encoder is frozen, so the pooled vectors of every pair are computed once, and
only the head is trained.
"""
import torch
from torch import nn


def document_pairs(doc):
    """Return the (head index, tail index) tensor of every ordered pair of a document."""
    n_entities = len(doc["entities"])
    return torch.tensor([(head, tail) for head in range(n_entities)
                         for tail in range(n_entities) if head != tail], dtype=torch.long)


@torch.no_grad()
def pair_inputs(doc_features, doc_pairs):
    """Return the head, tail and context vectors of the pairs of one document."""
    states = doc_features["states"].float()
    attention = doc_features["attention"].float()
    entities, entity_attention, row = [], [], 0
    for positions in doc_features["positions"]:
        entities.append(torch.logsumexp(states[positions], dim=0))
        entity_attention.append(attention[row:row + len(positions)].mean(0))
        row += len(positions)
    entities = torch.stack(entities)
    entity_attention = torch.stack(entity_attention)
    weights = entity_attention[doc_pairs[:, 0]] * entity_attention[doc_pairs[:, 1]]
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
    return entities[doc_pairs[:, 0]], entities[doc_pairs[:, 1]], weights @ states


def corpus_inputs(features, docs):
    """Return the pairs of each document and the (pairs, 3, hidden) float16 inputs of all pairs."""
    pairs = [document_pairs(doc) for doc in docs]
    inputs = torch.cat([torch.stack(pair_inputs(doc_features, doc_pairs), 1).half()
                        for doc_features, doc_pairs in zip(features, pairs)])
    return pairs, inputs


class PairClassifier(nn.Module):
    def __init__(self, n_relations, hidden=768, embedding=768, block=64, dropout=0.1):
        super().__init__()
        self.head = nn.Linear(2 * hidden, embedding)
        self.tail = nn.Linear(2 * hidden, embedding)
        self.bilinear = nn.Linear(embedding * block, n_relations + 1)
        self.dropout = nn.Dropout(dropout)
        self.block = block

    def forward(self, inputs):
        """Return the (pairs, relations + 1) logits of (pairs, 3, hidden) inputs."""
        heads, tails, contexts = self.dropout(inputs.float()).unbind(1)
        head = torch.tanh(self.head(torch.cat([heads, contexts], 1)))
        tail = torch.tanh(self.tail(torch.cat([tails, contexts], 1)))
        head = head.view(-1, head.shape[1] // self.block, self.block)
        tail = tail.view(-1, tail.shape[1] // self.block, self.block)
        grouped = (head.unsqueeze(3) * tail.unsqueeze(2)).view(head.shape[0], -1)
        return self.bilinear(grouped)


def adaptive_threshold_loss(logits, labels):
    """ATLOP loss: gold relations above the threshold class, the threshold above the rest.

    labels is a (pairs, relations + 1) 0/1 matrix whose column 0 marks pairs
    without a relation.
    """
    labels = labels.clone()
    labels[:, 0] = 0
    threshold = torch.zeros_like(labels)
    threshold[:, 0] = 1
    positive = labels + threshold
    negative = 1 - labels
    above = logits - (1 - positive) * 1e30
    below = logits - (1 - negative) * 1e30
    loss_above = -(torch.log_softmax(above, -1) * labels).sum(1)
    loss_below = -(torch.log_softmax(below, -1) * threshold).sum(1)
    return (loss_above + loss_below).mean()


def fact_scores(logits):
    """Return the (pairs, relations) scores sigmoid(logit - threshold logit).

    A score above 0.5 means the relation passes the learned threshold.
    """
    return torch.sigmoid(logits[:, 1:] - logits[:, :1])


def pair_labels(doc, pairs, relations):
    """Return the (pairs, relations + 1) label matrix of a document, column 0 for no relation."""
    column = {code: index + 1 for index, code in enumerate(relations)}
    row = {(int(head), int(tail)): index for index, (head, tail) in enumerate(pairs)}
    labels = torch.zeros(len(pairs), len(relations) + 1)
    for relation in doc["relations"]:
        labels[row[(relation["h"], relation["t"])], column[relation["r"]]] = 1
    labels[labels.sum(1) == 0, 0] = 1
    return labels


@torch.no_grad()
def predict(model, pairs, inputs, docs, relations, batch_size=4096, min_score=0.5):
    """Return {(document index, head id, tail id, relation code): score} for every scored fact."""
    model.eval()
    device = model.bilinear.weight.device
    scores = torch.cat([fact_scores(model(inputs[start:start + batch_size].to(device))).cpu()
                        for start in range(0, len(inputs), batch_size)])
    facts, offset = {}, 0
    for index, doc_pairs in enumerate(pairs):
        doc_scores = scores[offset:offset + len(doc_pairs)]
        offset += len(doc_pairs)
        ids = [entity["id"] for entity in docs[index]["entities"]]
        for row, column in zip(*torch.nonzero(doc_scores >= min_score, as_tuple=True)):
            head, tail = doc_pairs[row].tolist()
            facts[(index, ids[head], ids[tail], relations[column])] = float(doc_scores[row, column])
    return facts
