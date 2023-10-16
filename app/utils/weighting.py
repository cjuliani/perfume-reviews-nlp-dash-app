import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count = CountVectorizer()
matplotlib.use('agg')


def get_weights_per_word(
        corpus: List[str],
        words: Optional[List[str]],
        img_name: Optional[str] = None,
        show_fig: bool = True) -> Tuple[str, Dict[str, List[float]], List[Tuple[str, float]]]:
    """Returns weights per word from concepts."""
    # Get raw term counts
    word_count = count.fit_transform(
        raw_documents=corpus
    )

    # Convert raw term counts to TF-IDF
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count)

    # Get IDF weight per word
    df_idf = pd.DataFrame(
        tfidf_transformer.idf_,
        index=count.get_feature_names_out(),
        columns=["idf_weights"])

    results = {w: [] for w in words}
    total = 0.
    for w in words:
        results[w] = df_idf.loc[w].values[0]
        total += results[w]

    # Sort concept per weight
    weights = [item / total for _, item in results.items()]  # normalized
    weights_per_word = [(con, w) for con, w in zip(results.keys(), weights)]

    fig_path = display_bars(
        x=list(results.keys()),
        y=weights,
        img_name=img_name,
        show_fig=show_fig)

    return fig_path, results, weights_per_word


def display_bars(
        x: List[str],
        y: List[float],
        img_name: Optional[str] = None,
        show_fig: bool = True) -> Optional[str]:
    """Displays histogram of input values."""
    fig = Figure()
    ax = fig.add_subplot(111)

    # Create the bar plot using plt.bar
    ax.bar(x, y, width=0.7, color='grey', edgecolor='k', alpha=0.7)

    # Add labels and title
    ax.set_xlabel('Concept', fontsize=22, labelpad=10)
    ax.set_ylabel('Weight (normed)', fontsize=22, labelpad=10)

    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

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
