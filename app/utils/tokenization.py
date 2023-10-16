import string
import nltk
import enchant

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple, Dict, Any, Set

# Define english word dictionaries
dUS = enchant.Dict("en_US")
dGB = enchant.Dict("en_GB")

# Define stop words
nltk.data.path.append("./utils/data/nltk_data")
stop_words = set(stopwords.words('english'))

# Define lemmatizer and stop words
lemmatizer = WordNetLemmatizer()  # only for english language


def is_english(word: str) -> bool:
    """Returns True if input word within English dictionary."""
    return dGB.check(word) or dUS.check(word)


def get_wordnet_pos(treebank_tag: str) -> Any:
    """Returns tagging pos argument given word type."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def extract_tagged_tokens(
        text: str,
        with_stopwords: bool = True,
        english_only: bool = False) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """Returns token, tags and lemmatized words for given input text."""
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens if w not in list(string.punctuation)]  # punctuation removal
    if not with_stopwords:
        tokens = [w for w in tokens if w not in stop_words]  # stop words removal
    if english_only:
        tokens = [w for w in tokens if is_english(w) is True]  # non-English words removal

    # Perform tagging and lemmatization of words given their
    # respective tags.
    tagged = nltk.pos_tag(tokens)
    lemma_words = []
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:  # no tag supplying
            lemma = lemmatizer.lemmatize(word)
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
        lemma_words.append(lemma)

    return tokens, tagged, lemma_words


def process_tokens(
        texts: List[str],
        with_stopwords: bool,
        english_only: bool
) -> Tuple[List[List[str]], List[List[str]], List[List[Tuple[str, str]]], Dict[str, Dict[str, Set[int]]]]:
    """Returns tokens, lemmatized words and their respective tags,
    and the corpus of words indexed by texts."""
    tokens, lemma_words, tags = [], [], []
    corpus_dict = {}
    for i, text in enumerate(texts):
        # Collect tokens, tags and lemmatized words
        tok, tag, lemma = extract_tagged_tokens(text, with_stopwords, english_only)
        tokens.append(tok)
        lemma_words.append(lemma)
        tags.append(tag)

        # Collect sentences of current text
        sentences = sent_tokenize(text.lower())

        for j, word in enumerate(lemma):
            # Save review index and tag type for current
            # word. The review index is used to retrieve
            # the review associated to current word. A
            # sentiment analysis is performed on this review
            # later on to gain approximate insights on
            # word sentiments.
            if word not in corpus_dict:
                corpus_dict[word] = {
                    'review_index': dict(),
                    'tags': set()}

            # Add tag related to current word
            corpus_dict[word]['tags'].add(tag[j])

            # Get index of sentences from which current word exists
            sent_indices = [k for k, seq in enumerate(sentences) if tok[j] in seq]  # indices of sentences

            if not sent_indices:
                # Possibly a special character not to be counted
                continue

            if i in corpus_dict[word]['review_index']:
                # Do not associate review indices to current word
                # if already done for current text
                continue
            else:
                corpus_dict[word]['review_index'][i] = sent_indices  # associate word to review indices

        print(f'processed: {i + 1}/{len(texts)}', end='\r')

    return tokens, lemma_words, tags, corpus_dict
