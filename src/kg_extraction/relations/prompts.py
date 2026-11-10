"""Build the PromptORE context and entity names of a pair."""
from itertools import product

CONTEXTS = ("sentence", "mention_sentences", "document")


def entity_name(entity):
    """Return the longest mention of an entity."""
    return max((mention["name"] for mention in entity["mentions"]), key=len)


def shares_sentence(head, tail):
    """Return whether two entities have mentions in the same sentence."""
    return bool(_sentence_ids(head) & _sentence_ids(tail))


def build_context(doc, head, tail, context):
    """Return the context text of a pair, or None if the context does not exist.

    The sentence context only exists for entities that share a sentence.
    """
    sentences = doc["sents"]
    if context == "document":
        kept = range(len(sentences))
    elif context == "sentence":
        shared = _sentence_ids(head) & _sentence_ids(tail)
        if not shared:
            return None
        kept = [min(shared)]
    elif context == "mention_sentences":
        kept = _closest_mention_sentences(head, tail)
    else:
        raise ValueError(f"Unknown context: {context}")
    return " ".join(" ".join(sentences[index]) for index in kept)


def _sentence_ids(entity):
    return {mention["sent_id"] for mention in entity["mentions"]}


def _closest_mention_sentences(head, tail):
    """Return the sentences of the two closest mentions of the entities."""
    pairs = product(sorted(_sentence_ids(head)), sorted(_sentence_ids(tail)))
    first, second = min(pairs, key=lambda pair: (abs(pair[0] - pair[1]), pair))
    return sorted({first, second})
