print('Loading libraries...')
import os
import pickle
import concurrent.futures
import diskcache

from typing import Optional, Tuple, List, Dict, Union, Any
from dash import Input, Output, State, html, Dash
from dash.long_callback import DiskcacheLongCallbackManager
from layout import content_layout, get_files
from extraction import run_extraction
from translate import translate_texts, get_translator_model
from cleaning import process_cleaning_per_language, rule_out_duplicates_per_language

from utils.utils import generate_random_id
from utils.cleaning import correct_text_by_pattern, rule_out_by_size
from utils.sentiment import sentiment_analysis_by_concept
from utils.similarity import get_words_with_minimum_frequency, get_most_similar
from utils.analysis import select_data, word_frequency, analyze_sentiment_by_word, get_word_frequencies
from utils.analysis import retrieve_info_by_query, analyze_sentiment_reviews
from utils.weighting import get_weights_per_word
print('Done.')

cache = diskcache.Cache("./cache")
cache['extraction_progress'] = 'Initialization'
cache['translate_progress'] = 'Initialization'
cache['cleaning_progress'] = 'Initialization'
cache['words_frequency'] = []
cache['sent_analysis'] = []
cache['sent_analysis_word'] = []
cache['sent_analysis_concept_diagram'] = []
cache['sent_analysis_concept_weights'] = []
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Define the Dash app.
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
    long_callback_manager=long_callback_manager
)
server = app.server  # underlying flask server
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.layout = content_layout
app.config.suppress_callback_exceptions = True

# Pre-defined html styles
stand_style_1 = {"background-color": "#d1f0da", 'visibility': 'visible', 'padding': '10px'}
stand_style_2 = {'visibility': 'visible', 'padding': '10px'}
fig_style = {'visibility': 'visible'}
err_style = {"background-color": "#ffdbe2", 'visibility': 'visible', 'padding': '10px'}
last_style = {'visibility': 'hidden'}


@app.long_callback(
    output=[
        Output('output-extraction', 'children', allow_duplicate=True),
        Output('output-extraction', 'style'),
        Output('translate-dropdown-file', 'options', allow_duplicate=True)  # update dropdown of translate module
    ],
    inputs=Input('extraction-button', 'n_clicks'),
    running=[
        (Output("extraction-button", "disabled"), True, False),
        (Output("extraction-clock", "disabled"), False, True),
        (Output("extraction-button-cancel", "disabled"), False, True),
        (
            Output('output-extraction', 'style'),
            {"background-color": '#d1f4ff', 'visibility': 'visible', 'padding': '10px'},  # light-blue
            {"background-color": "#ffdbe2", 'visibility': 'visible', 'padding': '10px'},  # light-red
        ),
        (
            Output('output-extraction', 'children'),
            html.P("Processing..."),
            html.P("Cancelled.", style={'color': '#b83952'}),  # red
        ),
    ],
    cancel=[Input("extraction-button-cancel", "n_clicks")],
    state=[
        State('extraction-dropdown', 'value'),
        State('translate-dropdown-file', 'options')
    ]
)
def process_extraction(
        n_clicks: Optional[int],
        dropdown_value: Optional[str],
        dropdown_options: List[Dict[str, str]]) -> Tuple[html.P, Dict[str, str], List[Dict[str, str]]]:
    """Saves reviews extracted from original csv data as groups. Each
    group is defined by its respective language, which is detected per
    review."""
    if not dropdown_value:
        element = html.P("Please select a file.", style={'color': '#b83952'})
        return element, err_style, dropdown_options

    if n_clicks > 0:
        # Process data extraction
        results = None
        for item in run_extraction(dropdown_value):
            if type(item) == str:
                cache['extraction_progress'] = item
                continue

            results = item

        # After saving the pickle file, reload file options
        options = get_files('extraction')

        total = sum([len(results[lang]) for lang in results])
        element = html.P(f"File '{dropdown_value}' processed with {total} reviews.")

        return element, stand_style_1, options

    return html.P(""), last_style, dropdown_options


@app.long_callback(
    output=[
        Output('output-translate', 'children', allow_duplicate=True),
        Output('output-translate', 'style'),
        Output('cleaning-dropdown', 'options', allow_duplicate=True)
    ],
    inputs=Input('translate-button', 'n_clicks'),
    running=[
        (Output("translate-button", "disabled"), True, False),
        (Output("translate-clock", "disabled"), False, True),
        (Output("translate-button-cancel", "disabled"), False, True),
        (
                Output('output-translate', 'style'),
                {"background-color": '#d1f4ff', 'visibility': 'visible', 'padding': '10px'},  # light-blue
                {"background-color": "#ffdbe2", 'visibility': 'visible', 'padding': '10px'},  # light-red
        ),
        (
                Output('output-translate', 'children'),
                html.P("Processing..."),
                html.P("Cancelled.", style={'color': '#b83952'}),  # red
        )
    ],
    cancel=[Input("translate-button-cancel", "n_clicks")],
    state=[
        State('translate-dropdown-lang', 'value'),
        State('translate-dropdown-file', 'value'),
        State('cleaning-dropdown', 'options')
    ]
)
def process_translate(
        n_clicks: Optional[int],
        dropdown_value_1: Optional[str],
        dropdown_value_2: Optional[str],
        dropdown_options: List[Dict[str, str]]) -> Tuple[html.P, Dict[str, str], List[Dict[str, str]]]:
    """Translate texts to English given their associated language, which was
    previously detected during the extraction process."""
    if not dropdown_value_1 or not dropdown_value_2:
        element = html.P("Please select the language and the file to translate.", style={'color': '#b83952'})
        return element, err_style, dropdown_options

    if n_clicks > 0:
        file_name = dropdown_value_2.split('.')[0]
        data_name = file_name.split('_')[0]

        # Get translation and tokenizer models based on language
        # Note: these are not included in the threaded function
        # because threading would be stuck otherwise.
        cache['translate_progress'] = f"Loading translator for '{dropdown_value_1}'..."
        model, tokenizer = get_translator_model(input_language=dropdown_value_1)

        results = {}
        for item in translate_texts(data_name, dropdown_value_1, model, tokenizer):
            if type(item) == str:
                cache['translate_progress'] = item
            else:
                results = item

        # After saving pickle file, reload file options
        options = get_files('translate')

        total = sum([len(results[lang]) for lang in results])
        element = html.P(f"{total} reviews with language '{dropdown_value_1}' processed.")

        return element, stand_style_1, options

    return html.P(""), last_style, dropdown_options


@app.long_callback(
    output=[
        Output('output-cleaning', 'children', allow_duplicate=True),
        Output('output-cleaning', 'style'),
        Output('data-selection-dropdown', 'options', allow_duplicate=True)
    ],
    inputs=Input('cleaning-button', 'n_clicks'),
    running=[
        (Output("cleaning-button", "disabled"), True, False),
        (Output("cleaning-clock", "disabled"), False, True),
        (Output("cleaning-button-cancel", "disabled"), False, True),
        (
            Output('output-cleaning', 'style'),
            {"background-color": '#d1f4ff', 'visibility': 'visible', 'padding': '10px'},  # light-blue
            {"background-color": "#ffdbe2", 'visibility': 'visible', 'padding': '10px'},  # light-red
        ),
        (
            Output('output-cleaning', 'children'),
            html.P("Processing..."),
            html.P("Cancelled.", style={'color': '#b83952'}),  # red
        )
    ],
    cancel=Input("cleaning-button-cancel", "n_clicks"),
    state=[
        State('cleaning-dropdown', 'value'),
        State('cleaning-input', 'value'),
        State('data-selection-dropdown', 'options')
    ]
)
def process_cleaning(
        n_clicks: Optional[int],
        dropdown_value: Optional[str],
        input_value: int,
        dropdown_options: List[Dict[str, str]]) -> Tuple[html.P, Dict[str, str], List[Dict[str, str]]]:
    """Saves pickle file of English texts individually (1) cleaned from
    e.g., special characters, emoticons and punctuations, (2) corrected
    by converting e.g. word abbreviations or contractions to normal wording,
    and (3) ruled out if duplicated in the reviews."""
    if not dropdown_value:
        element = html.P("Please select the translated data to analyze.", style={'color': '#b83952'})
        return element, err_style, dropdown_options

    if not input_value or type(input_value) != int:
        element = html.P("Please provide an integer for minimum text length.", style={'color': '#b83952'})
        return element, err_style, dropdown_options

    if n_clicks > 0:
        file_name = dropdown_value.split('.')[0]
        language = file_name.split('_')[-1]
        data_name = file_name.split('_')[0]

        # Collect translated texts per language (dictionary)
        with open(f'processing/translated/{data_name}_{language}.pickle', 'rb') as file:
            translated = pickle.load(file)

        # Clean individual texts per language from
        # e.g., special characters, emoticons, time/date formats, etc.
        for item in process_cleaning_per_language(
            text_dict=translated,
            cleaning_function=correct_text_by_pattern,
            prefix_msg='pattern correction'):
            if type(item) == str:
                cache['cleaning_progress'] = item
                continue

            processed = item

        # Filter texts by minimum length
        # noinspection PyTypeChecker
        for item in process_cleaning_per_language(
            text_dict=processed,
            cleaning_function=rule_out_by_size,
            minimum_length=input_value,
            prefix_msg='text length filtering'):
            if type(item) == str:
                cache['cleaning_progress'] = item
                continue

            processed = item

        # Rule out duplicate comments i.e. possible spams
        # noinspection PyTypeChecker
        for item in rule_out_duplicates_per_language(
            text_dict=processed,
            prefix_msg='duplication removal'):
            if type(item) == str:
                cache['cleaning_progress'] = item
                continue

            processed = item

        # Save translated texts as pickle object.
        with open(f"processing/cleaned/{data_name}_{language}.pickle", 'wb') as file:
            pickle.dump(processed, file)

        # After saving pickle file, reload file options
        options = get_files('translate')
        element = html.P(f"File '{dropdown_value}' processed with {len(processed[language])} reviews.")
        print(f"File '{dropdown_value}' processed with {len(processed[language])} reviews.")
        return element, stand_style_1, options

    return html.P(""), last_style, dropdown_options


@app.callback(
    output=[
        Output('output-data-selection', 'children'),
        Output('output-data-selection', 'style')
    ],
    inputs=Input('data-selection-dropdown', 'value')
)
def process_data_selection(dropdown_data: Optional[str]) -> Tuple[html.P, Dict[str, str]]:
    """Saves processed data (tokens, tags, lemma words, etc.) in cache."""
    if dropdown_data:
        # If data selected has language name indicated,
        # keep dropdown menu for review language selection
        # disabled, and set the review language to analyze
        # to the one indicated in the data name.
        file_name = dropdown_data.split('.')[0]
        language = file_name.split('_')[-1]
        name = file_name.split('_')[0]

        executor = concurrent.futures.ThreadPoolExecutor()
        future = executor.submit(select_data, name, language)

        while not future.done():
            if future.result():
                (tokens, lemma_words, tags,
                 corpus_dict, cleaned) = future.result()

                # Save current data in cache for further usage in
                # e.g. sentiment analysis and word frequency calculation
                cache['tokens'] = tokens
                cache['lemma_words'] = lemma_words
                cache['tags'] = tags
                cache['corpus_dict'] = corpus_dict
                cache['cleaned'] = cleaned
                element = html.P(f"{len(tokens)} reviews processed with language '{language}'.")

                # Terminate the executor to release its resources
                executor.shutdown()

                return element, stand_style_2

    return html.P(""), last_style


@app.callback(
    output=[
        Output('output-frequency', 'children'),
        Output('output-frequency', 'style'),
        Output('output-frequency-img', 'children'),
    ],
    inputs=Input('analysis-freq-dropdown', 'value'),
    state=State('data-selection-dropdown', 'value')
)
def process_word_frequency(
        dropdown_value: Optional[str],
        dropdown_data: Optional[str]) -> Tuple[html.P, Dict[str, str], Optional[html.Img]]:
    """Returns figure plot of frequency analysis for words in
    review texts."""
    if not dropdown_data or cache['lemma_words'] is None:
        msg = f"Please select data to analyse."
        element = html.P(msg, style={'color': '#b83952'})
        return element, err_style, None

    input_words = [item for sublist in cache['lemma_words'] for item in sublist]
    img_id = generate_random_id(20)  # random image name

    fig_path = word_frequency(
        input_words=input_words,
        img_name=img_id,
        visualization_type=dropdown_value)

    element = html.P(f"{len(input_words)} words processed.")
    img = html.Img(src=fig_path, height='350px')

    # Clean out previous figures from assets
    clean_garbage('words_frequency', [fig_path])

    return element, stand_style_2, img


@app.callback(
    output=[
        Output('output-sent-analysis', 'children'),
        Output('output-sent-analysis', 'style'),
        Output('output-sent-analysis-img', 'children')
    ],
    inputs=Input('analysis-sent-dropdown', 'value'),
    state=State('data-selection-dropdown', 'value')
)
def process_sentiment_analysis(
        dropdown_value: Optional[str],
        dropdown_data: Optional[str]) -> Tuple[html.P, Dict[str, str], Optional[html.Img]]:
    """Returns figure plot of sentiment analysis results performed
    for all review texts in dataset."""
    if not dropdown_data:
        msg = f"Please select data to analyse and associated language first."
        element = html.P(msg, style={'color': '#b83952'})
        return element, err_style, None

    if not dropdown_value:
        element = html.P(f"Please select target sentiment to analyze.", style={'color': '#b83952'})
        return element, err_style, None

    else:
        file_name = dropdown_data.split('.')[0]
        language = file_name.split('_')[-1]
        img_id = generate_random_id(20)  # random image name

        fig_path = analyze_sentiment_reviews(
            data=cache['cleaned'],
            language=language,
            img_name=img_id,
            sentiment=dropdown_value)

        element = html.P(f"Sentiment analysis processed for {dropdown_value} reviews.")
        img = html.Img(src=fig_path, height='300px')

        # Clean out previous figures from assets
        clean_garbage('sent_analysis', [fig_path])

        return element, stand_style_2, img


@app.callback(
    output=[
        Output('output-sim-search', 'children', allow_duplicate=True),
        Output('output-sim-search', 'style')
    ],
    inputs=Input('sim-search-button', 'n_clicks'),
    state=[
        State('sim-input-text', 'value'),
        State('data-selection-dropdown', 'value')
    ]
)
def process_word_similarity(
        n_clicks: int,
        textarea_value: Optional[str],
        dropdown_data: Optional[str]) -> Tuple[Union[List[html.P], html.P], Dict[str, str]]:
    """Returns word that are semantically similar to the input word."""
    if not dropdown_data:
        element = html.P(f"Please select data to analyse.", style={'color': '#b83952'})
        return element, err_style

    if not textarea_value:
        element = html.P(f"Please type a target word to analyze.", style={'color': '#b83952'})
        return element, err_style

    if n_clicks > 0:
        executor = concurrent.futures.ThreadPoolExecutor()

        data = [item for sublist in cache['lemma_words'] for item in sublist]
        words, freqs = get_word_frequencies(words=list(data))
        frequent_words = get_words_with_minimum_frequency(words, freqs, min_number=10)

        future = executor.submit(get_most_similar, textarea_value, frequent_words, 15, None)

        while not future.done():
            if future.result():
                results = future.result()
                if len(results) > 1:
                    element = [
                        html.P("Similar words:"),
                        html.P(', '.join(results))
                    ]
                else:
                    element = [html.P("No similar words to show. Please, increase the number of reviews.")]

                # Terminate the executor to release its resources
                executor.shutdown()

                return element, stand_style_2

    return html.P(''), last_style


@app.callback(
    output=[
        Output('output-concept-msg', 'children'),
        Output('output-concept-diagram-img', 'children'),
        Output('output-concept-weights-img', 'children'),
        Output('output-concept-msg', 'style')
    ],
    inputs=Input('sent-diag-button', 'n_clicks'),
    state=[
        State('sent-input-concept', 'value'),
        State('data-selection-dropdown', 'value')
    ]
)
def process_sentiment_by_concept(
        n_clicks: int,
        textarea_value: Optional[str],
        dropdown_data: Optional[str]
) -> Tuple[Union[List[html.P], html.P], Optional[html.Img], Optional[html.Img], Dict[str, str]]:
    """Returns figure of sentiment analysis performed for a series
    of words belonging to a concept."""
    if not dropdown_data:
        element = html.P(f"Please select data to analyse.", style={'color': '#b83952'})
        return element, None, None, err_style

    if not textarea_value:
        msg = f"Please type concept words to analyze separated by comma."
        element = html.P(msg, style={'color': '#b83952'})
        return element, None, None, err_style

    if n_clicks > 0:
        file_name = dropdown_data.split('.')[0]
        language = file_name.split('_')[-1]

        # Split input words based on comma
        target_words = textarea_value.split(',')
        target_words = [word.strip() for word in target_words]

        img_id_1 = generate_random_id(20)  # random image name
        img_id_2 = generate_random_id(20)

        # Collect input words not existing in reviews
        non_existing = []
        for word in target_words:
            if word not in cache['corpus_dict']:
                non_existing.append(word)

        # If non-existent input words, returns error message
        if non_existing:
            msg = f"Words '{', '.join(non_existing)}' don't existing in the reviews."
            element = html.P(msg, style={'color': '#b83952'})
            return element, None, None, err_style
        else:
            # Generate figure plot for sentiment analysis
            fig_path_1, avg_scores = sentiment_analysis_by_concept(
                concept='main',
                reviews=cache['cleaned'],
                target_language=language,
                concept_dict={'main': target_words},
                corpus_dict=cache['corpus_dict'],
                score_thresh=0.25,
                windows_extraction=False,
                window_interval=5,
                img_name=img_id_1,
                show_fig=False)

            # Generate figure plot for word importance/weights
            fig_path_2, _, _ = get_weights_per_word(
                corpus=cache['cleaned'][language],
                words=target_words,
                img_name=img_id_2,
                show_fig=False)

            msg = [
                html.P(f"Average scores for {len(target_words)} words:"),
                html.P(f"  positive: {round(avg_scores[0], 2)}"),
                html.P(f"  neutral: {round(avg_scores[1], 2)}"),
                html.P(f"  negative: {round(avg_scores[2], 2)}")
            ]

            diagram_img = html.Img(src=fig_path_1, height='250px')
            weights_img = html.Img(src=fig_path_2, height='250px')

            # Clean out previous figures from assets
            clean_garbage('sent_analysis_concept_diagram', [fig_path_1])
            clean_garbage('sent_analysis_concept_weights', [fig_path_2])

            return msg, diagram_img, weights_img, stand_style_2

    return html.P(''), None, None, last_style


@app.callback(
    output=[
        Output('output-sent-word', 'children'),
        Output('output-sent-word', 'style'),
        Output('output-sent-word-img', 'children')
    ],
    inputs=Input('sent-viz-button', 'n_clicks'),
    state=[
        State('sent-input-word', 'value'),
        State('data-selection-dropdown', 'value')
    ]
)
def process_sentiment_by_word(
        n_clicks: int,
        textarea_value: Optional[str],
        dropdown_data: Optional[str]) -> Tuple[html.P, Dict[str, str], Optional[Union[html.Img, Any]]]:
    """Returns figure of sentiment scores related to input word."""
    if not dropdown_data:
        element = html.P(f"Please select data to analyse.", style={'color': '#b83952'})
        return element, err_style, None

    if not textarea_value:
        element = html.P(f"Please type a word to analyze.", style={'color': '#b83952'})
        return element, err_style, None

    if n_clicks > 0:
        file_name = dropdown_data.split('.')[0]
        language = file_name.split('_')[-1]
        img_id = generate_random_id(20)  # random image name

        try:
            # Display plot as jpg image generated by matplotlib
            fig_path = analyze_sentiment_by_word(
                word=textarea_value,
                data=cache['cleaned'],
                language=language,
                img_name=img_id,
                corpus_dict=cache['corpus_dict'])

            img = html.Img(src=fig_path, height='300px'),

            element = html.P(f"Sentiment analysis processed for word '{textarea_value}'.")

            # Clean out previous figures from assets
            clean_garbage('sent_analysis_word', [fig_path])

            return element, stand_style_2, img

        except KeyError:
            msg = f"Please provide a word existing in the reviews."
            element = html.P(msg, style={'color': '#b83952'})
            return element, err_style, None

    return html.P(''), last_style, None


@app.callback(
    output=[
        Output('output-retl-results', 'children'),
        Output('output-retl-results', 'style')
    ],
    inputs=Input('find-retl-button', 'n_clicks'),
    state=[
        State('retl-input-query', 'value'),
        State('data-selection-dropdown', 'value')
    ]
)
def process_information_retrieval(
        n_clicks: int,
        textarea_value: Optional[str],
        dropdown_data: Optional[str]) -> Tuple[Union[html.P, List[html.P]], Dict[str, str]]:
    """Returns review texts related to the input query."""
    if not dropdown_data:
        element = html.P(f"Please select data to analyse.", style={'color': '#b83952'})
        return element, err_style

    if not textarea_value:
        element = html.P(f"Please type a query to process.", style={'color': '#b83952'})
        return element, err_style

    if n_clicks > 0:
        data = cache['cleaned']
        file_name = dropdown_data.split('.')[0]
        language = file_name.split('_')[-1]

        results = retrieve_info_by_query(
            query=[textarea_value],
            texts=data[language])

        element = [html.P(f"Reviews:")] + [html.P('  > ' + str(item)) for item in results]
        return element, stand_style_2

    return html.P(''), last_style


def clean_garbage(service: str, new_items: List[str]):
    """Rule out useless images from assets."""
    items = cache[service]
    if len(items) > 1:
        os.remove(items[0])  # remove file locally
        items.remove(items[0])
    cache[service] = items + new_items


@app.callback(
    Output("output-extraction", "children"),
    Input("extraction-clock", "n_intervals"))
def extraction_update(_):
    """Returns progress message to extraction output Div."""
    new_value = cache['extraction_progress']
    return html.P(new_value)


@app.callback(
    Output("output-translate", "children"),
    Input("translate-clock", "n_intervals"))
def translate_update(_):
    """Returns progress message to translate output Div."""
    new_value = cache['translate_progress']
    return html.P(new_value)


@app.callback(
    Output("output-cleaning", "children"),
    Input("cleaning-clock", "n_intervals"))
def cleaning_update(_):
    """Returns progress message to cleaning output Div."""
    new_value = cache['cleaning_progress']
    return html.P(new_value)


if __name__ == '__main__':
    app.run_server(
        port=8080,
        host="0.0.0.0",
        dev_tools_hot_reload=False,
        debug=True,
        use_reloader=False  # avoid loading twice in debug mode
    )
