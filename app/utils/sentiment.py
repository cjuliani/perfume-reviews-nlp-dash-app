import os
import numpy as np
import matplotlib
import textblob
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Optional, Union
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from nltk.tokenize import sent_tokenize, word_tokenize


matplotlib.use('agg')


def sentiment_analysis_by_word(
        target_word: str,
        corpus_dict: Dict[str, Dict[str, set]],
        reviews: Dict[str, List],
        target_language: str,
        score_thresh: float = 0.35,
        windows_extraction: bool = False,
        window_interval: int = 5
) -> Tuple[Dict[str, Dict[str, List[Union[float, str]]]], float, float, float, float]:
    """Returns the sentiment analysis values."""
    # Collect texts associated to target word.
    review_indices = list(corpus_dict[target_word]['review_index'])

    results_dict = {
        'scores': {'pos': [], 'neu': [], 'neg': [], 'total': []},
        'texts': {'pos': [], 'neu': [], 'neg': []}
    }
    for i in review_indices:
        # Define sentences from review
        sentences = sent_tokenize(reviews[target_language][i])

        # Define sentences related to target word
        sentence_indices = corpus_dict[target_word]['review_index'][i]
        sentences = [sentences[i] for i in sentence_indices]

        # Collect text by window
        if windows_extraction:
            sentences = extract_sentences_by_window(
                target_word=target_word,
                input_sentences=sentences,
                window_interval=window_interval)

        for text in sentences:
            # Calculate sentiment polarity
            testimonial = textblob.TextBlob(text)
            score = testimonial.sentiment.polarity

            results_dict['scores']['total'].append(score)

            if score >= score_thresh:
                # positive sentiment closer to 1.
                results_dict['scores']['pos'].append(score)
                results_dict['texts']['pos'].append(text)
            elif -score_thresh <= score < score_thresh:
                results_dict['scores']['neu'].append(score)
                results_dict['texts']['neu'].append(text)
            else:
                # negative sentiment closer to -1.
                results_dict['scores']['neg'].append(score)
                results_dict['texts']['neg'].append(text)

    # Calculate mean scores
    total_scores = (results_dict['scores']['pos'] +
                    results_dict['scores']['neu'] +
                    results_dict['scores']['neg'])

    avg_scores = round(sum(total_scores) / len(total_scores), 2)
    avg_pos = round(len(results_dict['scores']['pos']) / len(total_scores), 2)
    avg_neut = round(len(results_dict['scores']['neu']) / len(total_scores), 2)
    avg_neg = round(len(results_dict['scores']['neg']) / len(total_scores), 2)

    return results_dict, avg_scores, avg_pos, avg_neut, avg_neg


def repl(input_words: List[str], indices: List[int], window_interval: int) -> List[str]:
    combined_words = []
    for i in indices:
        start, end = max(i - window_interval, 0), i + window_interval
        combined_words.append(' '.join(input_words[start:end]))
    return combined_words


def extract_sentences_by_window(
        target_word: str,
        input_sentences: List[str],
        window_interval: int = 5) -> List[str]:
    """Returns a fraction of sentence or text given an input target
    word. This fraction is a sequence of words collected given a
    window interval centered at the target word."""
    output_sentences = []
    for sent in input_sentences:
        words = word_tokenize(sent)
        target_word_index = [i for i, word in enumerate(words) if word.lower().startswith(target_word)]

        results = repl(words, target_word_index, window_interval)
        output_sentences += results

    return output_sentences


def display_histogram(
        values: List[float],
        img_name: Optional[str] = None,
        bins: Optional[int] = None,
        show_fig: bool = True) -> Optional[str]:
    """Displays histogram of input values."""
    if bins is None:
        bins = np.arange(-1, 1.1, 0.1)  # distribute bins at 0.1 intervals

    # Compute the histogram and normalize frequencies
    hist, bin_edges = np.histogram(values, bins, density=True)
    normed = hist / np.sum(hist)

    fig = Figure()
    ax = fig.add_subplot(111)

    # Create the bar plot using plt.bar
    ax.bar(bin_edges[:-1], normed, width=0.07, color='grey', edgecolor='k', alpha=0.7)

    # Add labels and title
    ax.set_xlabel('Sentiment Score', fontsize=22, labelpad=10)
    ax.set_ylabel('Frequency', fontsize=22, labelpad=10)

    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    if bins is None:
        ax.set_xticks(np.arange(-1, 1.1, 0.1))

    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_position(('outward', 2))
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_position(('outward', 8))

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


def sentiment_analysis_by_concept(
        concept: str,
        corpus_dict: Dict[str, Dict[str, set]],
        concept_dict: Dict[str, List[str]],
        target_language: str,
        reviews: Dict[str, List],
        score_thresh: float = 0.35,
        windows_extraction: bool = False,
        window_interval: int = 5,
        img_name: Optional[str] = None,
        show_fig: bool = True) -> Tuple[Optional[str], List[float]]:
    """Returns sentiment scores associated to given concept and
    display sentiment results for each word related to that
    concept."""
    # Collect words making up current concept
    concept_words = concept_dict[concept]

    # Define sentiment categories
    category_names = ['positive', 'neutral', 'negative']

    total_scores = []
    results = {}
    concept_avg_scores = np.zeros(3)
    for word in concept_words:
        # Get sentiment scores for current concept word
        tmp = sentiment_analysis_by_word(
            target_word=word,
            corpus_dict=corpus_dict,
            reviews=reviews,
            target_language=target_language,
            score_thresh=score_thresh,
            windows_extraction=windows_extraction,
            window_interval=window_interval
        )

        # Store results by target word
        results[word] = [*tmp[2:]]
        total_scores += tmp[0]

        # Store total results
        concept_avg_scores += np.array([*tmp[2:]]) / len(concept_words)

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn_r'](
        np.linspace(0.15, 0.85, data.shape[1]))

    # ---
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    # noinspection PyArgumentList
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(
            labels, widths,
            left=starts,
            height=0.5,
            label=colname,
            color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color, fontsize=22)

    ax.tick_params(axis='y', labelsize=22)  # Increase tick label font size on the y-axis
    ax.yaxis.set_tick_params(width=2.5)

    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.legend(
        ncols=len(category_names), bbox_to_anchor=(0, 1),
        loc='lower left', fontsize=22)

    if show_fig:
        plt.show()
        return None, list(concept_avg_scores)
    else:
        # Delete existing files
        fig_path = f"assets/{img_name}.jpg"
        if os.path.exists(fig_path):
            os.remove(fig_path)

        # Save plot figure in assets
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig_path, list(concept_avg_scores)


def sentiment_analysis_by_text(
        texts: Union[List[str], List[List[str]]],
        score_thresh: float = 0.35) -> Tuple[List[float], Dict[str, Dict[str, List[Union[float, str]]]]]:
    """Returns sentiment scores associated to given texts and
    display sentiment results for each word related to that
    concept."""
    if type(texts[0]) == list:
        texts = [item for sublist in texts for item in sublist]

    results_dict = {
        'scores': {'pos': [], 'neut': [], 'neg': [], 'total': []},
        'texts': {'pos': [], 'neut': [], 'neg': []}
    }
    for text in texts:
        # Calculate sentiment polarity
        testimonial = textblob.TextBlob(text)
        score = testimonial.sentiment.polarity

        results_dict['scores']['total'].append(score)

        if score >= score_thresh:
            # positive sentiment closer to 1.
            results_dict['scores']['pos'].append(score)
            results_dict['texts']['pos'].append(text)
        elif -score_thresh <= score < score_thresh:
            results_dict['scores']['neut'].append(score)
            results_dict['texts']['neut'].append(text)
        else:
            # negative sentiment closer to -1.
            results_dict['scores']['neg'].append(score)
            results_dict['texts']['neg'].append(text)

    # Calculate mean scores
    total_scores = (results_dict['scores']['pos'] +
                    results_dict['scores']['neut'] +
                    results_dict['scores']['neg'])

    return total_scores, results_dict
