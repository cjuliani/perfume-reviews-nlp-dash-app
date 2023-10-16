import os

from dash import dcc, html
from typing import Dict, List


# Define corpus of language available in reviews
# which need to be translated to English. For example,
# if 'French' is selected, only French reviews will be
# translated
language_corpus = (
    'english',
    'arabic', 'russian', 'french', 'german',
    'dutch', 'italian', 'chinese', 'portuguese'
)

sentiment_options = ['general', 'positive', "neutral", 'negative']


def get_files(name: str) -> List[Dict[str, str]]:
    """Returns file names collected from specified folder name."""
    if name == 'cleaning':
        folder_path = 'processing/cleaned'
    elif name == 'translate':
        folder_path = 'processing/translated'
    elif name == 'extraction':
        folder_path = 'processing/not_translated'
    elif name == 'data':
        folder_path = 'data'
    else:
        folder_path = ''
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files = [{'label': file_name.split('.')[0], 'value': file_name} for file_name in files]
    return files


content_layout = html.Div([
    html.Div(
        id="banner",
        children=[
            html.H1('Perfume Reviews Analysis')
        ]
    ),

    html.Div(
        className="flex-container",  # Added a class name for the flex container
        children=[
            html.Div(
                id='left-column',
                children=[
                    # --- EXTRACT
                    html.H3("Extraction"),
                    html.Label('Select data to extract:'),
                    dcc.Dropdown(
                        id='extraction-dropdown',
                        className='dropdown',
                        options=get_files('data'),
                        value=None  # Default value is None
                    ),
                    html.Button('Run', id='extraction-button', className='run-button', n_clicks=0),
                    html.Button('Cancel', id="extraction-button-cancel", className='run-button', n_clicks=0),
                    dcc.Interval(
                        id='extraction-clock',
                        disabled=True,
                        interval=250,
                        n_intervals=0,
                        max_intervals=-1),
                    html.Div(
                        id='output-extraction',
                        className='data-processing'
                    ),

                    # --- TRANSLATE
                    html.H3("Translate to English"),
                    html.Label('Select input language to translate from:'),
                    dcc.Dropdown(
                        id='translate-dropdown-lang',
                        className='dropdown',
                        options=[{'label': lang, 'value': lang} for lang in language_corpus],
                        value=None  # Default value is None
                    ),
                    html.Label('Select file to translate:'),
                    dcc.Dropdown(
                        id='translate-dropdown-file',
                        className='dropdown',
                        options=get_files('extraction'),
                        value=None  # Default value is None
                    ),
                    html.Button('Run', id='translate-button', className='run-button', n_clicks=0),
                    html.Button('Cancel', id="translate-button-cancel", className='run-button', n_clicks=0),
                    dcc.Interval(
                        id='translate-clock',
                        disabled=True,
                        interval=250,
                        n_intervals=0,
                        max_intervals=-1),
                    html.Div(
                        id='output-translate',
                        className='data-processing',
                    ),

                    # --- CLEANING
                    html.H3("Cleaning"),
                    html.Label('Select translated data to clean:'),
                    dcc.Dropdown(
                        id='cleaning-dropdown',
                        className='dropdown',
                        options=get_files('translate'),
                        value=None  # Default value is None
                    ),
                    html.Label('Select minimum text review length:'),
                    dcc.Input(id='cleaning-input', value=15, type='number'),
                    html.Div([
                        html.Button('Run', id='cleaning-button', className='run-button', n_clicks=0),
                        html.Button('Cancel', id="cleaning-button-cancel", className='run-button', n_clicks=0),
                    ], className='cleaning-block'),
                    dcc.Interval(
                        id='cleaning-clock',
                        disabled=True,
                        interval=250,
                        n_intervals=0,
                        max_intervals=-1),
                    html.Div(
                        id='output-cleaning',
                        className='data-processing',
                    )
                ]),

            html.Div(
                id="right-column",
                children=[
                    # --- Words Frequency
                    html.H3("Data selection"),
                    html.Label('Select data to analyse:'),
                    dcc.Dropdown(
                        id='data-selection-dropdown',
                        className='dropdown',
                        options=get_files('cleaning'),
                        value=None  # Default value is None
                    ),
                    html.Div(
                        id='output-data-selection',
                        className='data-processing'
                    ),

                    html.H3("Words Frequency"),
                    html.Label('Select visualization:'),
                    dcc.Dropdown(
                        id='analysis-freq-dropdown',
                        className='dropdown',
                        options=[{'label': name, 'value': name} for name in ['histogram', "word cloud"]],
                        value=None  # Default value is None
                    ),
                    html.Div(
                        id='output-frequency',
                        className='data-processing'
                    ),
                    html.Div(id='output-frequency-img'),

                    # --- Sentiment Analysis
                    html.H3("Sentiment Analysis"),
                    html.Label('Select target sentiment:'),
                    dcc.Dropdown(
                        id='analysis-sent-dropdown',
                        className='dropdown',
                        options=[{'label': name, 'value': name} for name in sentiment_options],
                        value=None  # Default value is None
                    ),
                    html.Div(
                        id='output-sent-analysis',
                        className='data-processing'
                    ),
                    html.Div(id='output-sent-analysis-img'),

                    html.Div([
                        html.Label('Find words with semantic similarity to target word:'),
                        html.Div([
                            dcc.Input(
                                id='sim-input-text',
                                value=None,
                                placeholder='Enter one word (example: perfume)',
                                type='text'),
                            html.Button('Search', id='sim-search-button', n_clicks=0),
                        ], className='container'),
                        html.Div(
                            id='output-sim-search',
                            className='data-processing'
                        ),
                    ], className='sent-block'),

                    html.Div([
                        html.Label('Visualize sentiment by word:'),
                        html.Div([
                            dcc.Input(
                                id='sent-input-word',
                                value=None,
                                placeholder='Enter one word (example: smell)',
                                type='text'),
                            html.Button('Display', id='sent-viz-button', n_clicks=0),
                        ], className='container'),
                        html.Div(
                            id='output-sent-word',
                            className='data-processing'
                        ),
                        html.Div(id='output-sent-word-img'),
                    ], className='sent-block'),

                    html.Div([
                        html.Label('Visualize sentiment scores by concept:'),
                        html.Div([
                            dcc.Input(
                                id='sent-input-concept',
                                value=None,
                                placeholder='Enter words separated by a comma (e.g. sweet, smell, ...)',
                                type='text'),
                            html.Button('Display', id='sent-diag-button', n_clicks=0),
                        ], className='container'),
                        html.Div(
                            id='output-concept-msg',
                            className='data-processing'
                        ),
                        html.Div(id='output-concept-diagram-img'),
                        html.Div(id='output-concept-weights-img'),
                    ], className='sent-block'),

                    # --- Information Retrieval
                    html.H3("Information Retrieval"),
                    html.Label('Find reviews related to query:'),
                    html.Div([
                        dcc.Input(
                            id='retl-input-query',
                            value=None,
                            placeholder='Enter query word(s) (example: product)',
                            type='text'),
                        html.Button('Search', id='find-retl-button', n_clicks=0),
                    ], className='container'),
                    html.Div(
                        id='output-retl-results',
                        className='data-processing'
                    )
                ])
        ]
    )
], style={'margin': 'auto'})
