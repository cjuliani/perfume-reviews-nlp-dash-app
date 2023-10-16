import os
import nltk
import string
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .tokenization import process_tokens
from .sentiment import sentiment_analysis_by_word
from .sentiment import display_histogram, sentiment_analysis_by_text

from wordcloud import WordCloud
from collections import Counter
from typing import Optional, List, Dict, Tuple, Set, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

matplotlib.use('agg')

# Define stop words
nltk.data.path.append("./utils/data/nltk_data")
stop_words = set(nltk.corpus.stopwords.words('english'))


def select_data(
        name: str,
        language: str
) -> Tuple[List[List[str]], List[List[str]], List[List[Tuple[str, str]]], Dict[str, Dict[str, Set[int]]], Any]:
    """Returns tokens, tags and lemmatized words after processing texts
    of given language."""
    # Collect cleaned texts per language (dictionary)
    with open(f'processing/cleaned/{name}_{language}.pickle', 'rb') as file:
        cleaned = pickle.load(file)

    # Collect tokens, lemmatized words and tags from review.
    # The corpus dictionary associates words to respective
    # review texts (by indexing) - i.e. lemma words.
    # This function also rules out emojis and puntuations.
    tokens, lemmatized_words, tags, corpus_dict = process_tokens(
        cleaned[language],
        with_stopwords=False,  # if True, rule out stop words
        english_only=False  # if True, consider words present in English dictionary only
    )

    return tokens, lemmatized_words, tags, corpus_dict, cleaned


def word_frequency(
        input_words: List[str],
        img_name: str,
        visualization_type: str = 'histogram') -> str:
    """Saves figure plot of words frequency and returns the address
    location of locally saved image."""
    words, freqs = get_word_frequencies(input_words)

    if visualization_type == 'histogram':
        # Display words frequency, selected by indices (within interval).
        fig_path = check_words_frequency(
            words, freqs,
            incr=0,  # starts word collection from index 0
            intv=25,  # keep Y words (from interval) out of X words
            img_name=img_name,
            show_fig=False
        )
    else:
        # Display word cloud from lemmatized or tagged cord corpus
        fig_path = check_word_cloud(
            words=input_words,
            tag_prefix=None,
            img_name=img_name,
            show_fig=False
        )

    return fig_path


def analyze_sentiment_reviews(
        data: Dict[str, List],
        language: str,
        img_name: str,
        sentiment: str = 'general') -> str:
    """Saves figure plot of scores distribution for given sentiment."""
    # Analyze sentiment from texts
    total_scores, results_dict = sentiment_analysis_by_text(
        texts=data[language],
        score_thresh=0.35
    )

    if sentiment == 'general':
        fig_path = display_histogram(
            values=total_scores,
            img_name=img_name,
            bins=None, show_fig=False)
    elif sentiment == 'positive':
        fig_path = display_histogram(
            values=results_dict['scores']['pos'],
            img_name=img_name,
            bins=None, show_fig=False)
    elif sentiment == 'neutral':
        fig_path = display_histogram(
            values=results_dict['scores']['neut'],
            img_name=img_name,
            bins=None, show_fig=False)
    else:
        fig_path = display_histogram(
            values=results_dict['scores']['neg'],
            img_name=img_name,
            bins=None, show_fig=False)
    return fig_path


def analyze_sentiment_by_word(
        word: str,
        data: Dict[str, List[str]],
        language: str,
        corpus_dict: Dict[str, Dict[str, Set[str]]],
        img_name: str) -> str:
    """Saves figure plot of sentiment scores calculated for given word."""
    results = sentiment_analysis_by_word(
        target_word=word,
        corpus_dict=corpus_dict,
        reviews=data,
        target_language=language,
        score_thresh=0.35,
        windows_extraction=False,
        window_interval=5
    )
    fig_path = display_histogram(
        values=results[0]['scores']['total'],
        img_name=img_name,
        bins=None,
        show_fig=False)

    return fig_path


def retrieve_info_by_query(query: List[str], texts: List[str]) -> List[str]:
    """Returns reviews relevant to input query."""
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Transform the documents into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # Perform cosine similarity search
    query_vector = tfidf_vectorizer.transform(query)  # creates a (m,) vector, where m is number of unique words

    # Calculate distance between this (m,) vector and other (n, m) vectors
    cosine_similarities = np.squeeze(
        linear_kernel(query_vector, tfidf_matrix))  # returns: (n,) where n = number documents
    indices = np.argsort(cosine_similarities)[::-1]  # descending order

    # Returns 10 most relevant query-related reviews.
    return [texts[i] for i in indices[:10]]


def get_word_frequencies(
        words: Optional[List[str]] = None,
        texts: Optional[List[str]] = None) -> Tuple[List, List]:
    """Returns lists of word frequencies and words ordered
    by frequencies (in descending order)."""
    if words and not texts:
        if type(words[0]) == list:
            words_selection = [item for sublist in words for item in sublist]  # flattening
        else:
            words_selection = words
    elif texts and not words:
        words_selection = []
        if type(texts[0]) == list:
            texts = [item for sublist in texts for item in sublist]
        for item in texts:
            words_selection += nltk.word_tokenize(item)
    else:
        raise Exception("Words and texts can't be provided simultaneously. Choose one.")

    # Cleaning
    words_selection = [word for word in words_selection if word.lower() not in stop_words]
    words_selection = [word for word in words_selection if word.lower() not in list(string.punctuation)]

    # Calculate word frequencies for current selection (tag)
    counts = Counter(words_selection)
    word_frequencies = list(counts.keys())
    value_frequencies = [int(val) for val in list(counts.values())]

    # Sorting
    indices = np.argsort(np.array(value_frequencies))[::-1]
    words = [word_frequencies[i] for i in indices]
    freqs = [value_frequencies[i] for i in indices]

    return words, freqs


def check_words_frequency(
        words: List,
        freqs: List,
        incr: int = 0,
        intv: int = 15,
        img_name: Optional[str] = None,
        show_fig: bool = True) -> Optional[str]:
    """Saves word frequency plot as bar chart."""
    fig = Figure()
    ax = fig.add_subplot(111)

    ax.barh(words[incr * intv:intv + (intv * incr)][::-1], freqs[incr * intv:intv + (intv * incr)][::-1])
    ax.set_xlabel('Frequency', fontsize=14)
    ax.set_ylabel('Words', fontsize=14)
    ax.set_title('Concept Frequencies Based on Tagged Words', fontsize=14)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    if show_fig:
        plt.show(block=True)
        return None
    else:
        # Delete existing files
        fig_path = f"assets/{img_name}.jpg"
        if os.path.exists(fig_path):
            os.remove(fig_path)

        # Save plot figure in assets
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig_path


def check_word_cloud(
        words: List[str],
        tag_prefix: Optional[str] = None,
        img_name: Optional[str] = None,
        show_fig: bool = True) -> Optional[str]:
    """Saves word cloud for given word list."""
    if tag_prefix is not None:
        try:
            df = pd.DataFrame(words, columns=['value', 'code'])
            words = df[df['code'].apply(lambda x: x.startswith(tag_prefix))].to_numpy()[:, 0].tolist()
        except ValueError:
            raise Exception('You must provide tags in format (1, 2) per item.')

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white').generate(' '.join(words))

    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  # Turn off axis labels and ticks

    if show_fig:
        plt.show(block=True)
    else:
        # Delete existing files
        fig_path = f"assets/{img_name}.jpg"
        if os.path.exists(fig_path):
            os.remove(fig_path)

        # Save plot figure in assets
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig_path
