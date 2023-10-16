import spacy
import numpy as np

from queue import Queue
from typing import List

# Load English tokenizer, tagger, NER (large)
nlp = spacy.load("./utils/data/spacy_data/en_core_web_lg/en_core_web_lg-3.6.0")


def get_words_with_minimum_frequency(
        words: List[str],  # must be sorted given frequencies
        freqs: List[float],  # must be sorted in descending order
        min_number: int = 10) -> List[str]:
    """Returns list of words with frequency exceeding
    given minimum."""
    cond = [i < min_number for i in freqs]
    index = cond.index(True)
    return words[:index]


def get_most_similar(
        target_word: str,
        corpus: List[str],
        top: int = 10,
        queue_obj: Queue = None) -> List[str]:
    """Returns top N words of corpus whose similarity with
    target word is high."""
    values = []
    for i, word in enumerate(corpus):
        similarity = nlp(target_word).similarity(nlp(word))
        values.append(similarity)

        msg = f'step {i}/{len(corpus)}'
        print(msg, end='\r')

        if queue_obj is not None:
            queue_obj.put(msg)

    indices_high_to_low = np.argsort(values)[::-1].tolist()

    return [corpus[i] for i in indices_high_to_low[:top]]
