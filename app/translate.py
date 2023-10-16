import re
import pickle

from typing import Any, Dict, List, Union
from utils.cleaning import rule_out_extra_ending


def find_length(n: int) -> int:
    for number in [16, 32, 64, 128, 256, 512]:
        if number > n:
            return number
    return 512


def decode(
        input_text: str,
        text_language: str,
        translator: Any,
        tokenizer_model: Any,
        input_length: int) -> str:
    """
    Returns translated input text using the MarianMT (Machine
    Translation) translator.
    """
    input_ids = tokenizer_model.encode(
        input_text, text_language,
        return_tensors="pt"  # among 'pt', 'tf', 'np', or 'jax'
    )

    translation = translator.generate(
        input_ids,
        max_length=input_length,
        num_return_sequences=1,  # number of input texts to decode
        num_beams=4,
        early_stopping=True)

    return tokenizer_model.decode(
        translation[0],
        skip_special_tokens=True)


def translate_text(
        review: str,
        text_language: str,
        translator: Any,
        tokenizer_model: Any) -> str:
    """
    Returns translated review for given text_language.
    """
    text_length = len(review)

    # Define base inputs with known parameters
    base_inputs = (text_language, translator, tokenizer_model, find_length(text_length))

    if text_length > 512:
        # Split the review into different sentences if too
        # long for processing by the MarianMTModel translator.
        if text_language.startswith('chines'):
            sentences = review.split('。')
        else:
            sentences = re.split(r'(?<=[.!?])\s+', review)  # latin

        # Limit each sentence to 512 characters
        sentences = [seq[:512] for seq in sentences]

        results = ''
        for seq in sentences:
            # noinspection PyTypeChecker
            text = decode(*(seq,) + base_inputs)
            results += text + ' '
        return rule_out_extra_ending(results[:-1])  # rule out last white space
    else:
        # noinspection PyTypeChecker
        output_text = decode(*(review,) + base_inputs)
        return rule_out_extra_ending(output_text)


def get_translator_model(input_language='english'):
    """Returns translator translator_model and mt_tokenizer given input input_language."""
    if input_language != 'english':
        if input_language == 'german':
            path = 'models/models_Helsinki_NLP_opus_mt_de_en/snapshots/1a922f3b32a8e809e17a47d4b32142d8105924e5'
        elif input_language == 'italian':
            path = 'models/models_Helsinki_NLP_opus_mt_it_en/snapshots/42556a0848fc726f4d27399f20b19ff6f01afe11'
        elif input_language == 'portuguese':
            path = 'models/models_Helsinki_NLP_opus_mt_roa_en/snapshots/c26705dd23ff67aa0019dbde41e9456c3a0900ef'
        elif input_language == 'spanish':
            path = 'models/models_Helsinki_NLP_opus_mt_roa_en/snapshots/c26705dd23ff67aa0019dbde41e9456c3a0900ef'
        elif input_language.startswith('chines'):
            path = 'models/models_Helsinki_NLP_opus_mt_zh_en/snapshots/cf109095479db38d6df799875e34039d4938aaa6'
        elif input_language == 'arabic':
            path = 'models/models_Helsinki_NLP_opus_mt_tc_big_ar_en/snapshots/bcb4acd39ee8e3552e171653a8e31a10729b4330'
        elif input_language == 'french':
            path = 'models/models_Helsinki-NLP_opus_mt_fr_en/snapshots/b4a9a384c2ec68a224bbd2ee3fd5df0c71ca5b1b'
        elif input_language == 'dutch':
            path = 'models/models_Helsinki_NLP_opus_mt_nl_en/snapshots/48af999f2c59b10c05ca6e008dcedc07677a9b15'
        elif input_language == 'russian':
            path = 'models/models_Helsinki_NLP_opus_mt_ru_en/snapshots/fbd6dc73284f95536648512cc21d57f19191961a'
        else:
            raise Exception(f"Language '{input_language}' not supported for translation.")

        # Load translator_model
        from transformers import MarianTokenizer, MarianMTModel
        # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html
        mt_model = MarianMTModel.from_pretrained(path, local_files_only=True)
        mt_tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)

        return mt_model, mt_tokenizer
    else:
        return None, None  # no translator needed for english


def translate_texts(
        data_name: str,
        language: str,
        model: Any,
        tokenizer: Any) -> Union[str, Dict[str, List[str]]]:
    """Saves review texts translated to English given associated language
    corpus (previously detected)."""
    translated_texts = {language: []}

    # Collect dictionary of texts per text_language not translated
    with open(f'processing/not_translated/{data_name}_not_translated.pickle', 'rb') as file:
        texts_to_translate = pickle.load(file)

    # Collect texts to be translated for current text_language.
    reviews = texts_to_translate[language]

    if language != 'english':
        # Translate texts in english if current text_language not english.
        # We use multiprocessing, which keeps original data order.
        for i, text in enumerate(reviews):
            trans = translate_text(
                review=text,
                text_language=language,
                translator=model,
                tokenizer_model=tokenizer
            )
            translated_texts[language].append(trans)

            yield f"step {i + 1}/{len(reviews)}"

    else:
        print(f"Already translated to English.")
        translated_texts['english'] = reviews

    with open(f"processing/translated/{data_name}_{language}.pickle", 'wb') as file:
        pickle.dump(translated_texts, file)

    yield translated_texts
