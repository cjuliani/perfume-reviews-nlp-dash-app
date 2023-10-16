import re
import nltk
import difflib

from nltk.corpus import words
from .word_utils import cont_common, cont_abbreviations, cont_time, language_corpus
from typing import Dict, Any, Sequence, Optional, Union

# Define corpus of English words
nltk.data.path.append("./utils/data/nltk_data")
english_corpus = words.words()


def repl(match: Any, min_threshold: float = 0.9) -> str:
    """Returns best matching word given input minimum
    similarity threshold."""
    converted_word = match.group(1)
    return find_best_match(converted_word, min_threshold)


def similarity_score(a: str, b: str) -> float:
    """Returns the similarity score between input a and b."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_best_match(input_word: str, min_threshold: float = 0.9) -> Union[Sequence[Any], str]:
    """Returns best matching English word for given input."""
    closest_matches = difflib.get_close_matches(input_word, english_corpus)

    if closest_matches:
        # Check that matching word is very similar to reference
        # noinspection PyTypeChecker
        sim = similarity_score(closest_matches[0], input_word)
        if sim >= min_threshold:
            return closest_matches[0]
        else:
            return input_word
    else:
        return input_word  # return original if not matches


def correct_text_by_pattern(text: str, **kwargs: Dict[str, str]) -> str:
    """Returns input text cleaned from emoticons, time indicators, urls,
    html codes, punctuations, special and repeated characters, unicodes,
    word cont_common, and consecutive points.
    """
    # Remove emoticons
    emoticons_to_remove = [
        ":P", ":-P", ":-D", ":D", ":o",
        ":O", ":/", r":\\/", r":\\", r":-\\",
        ":|", ":-|", ":-o", ":x", ":-x",
        ";)", ";-)", ":)", ":-)", ":(",
        ":-(", "<3", "^^", "^_^", "^_*",
        "*_^", "*_*", "-_-", ":X", ":x",
        "=)", ":0)", ":o)", ":O)"]
    pattern = "|".join(map(re.escape, emoticons_to_remove))
    output_text = re.sub(pattern, "", text)

    # Remove time indicator for each comment (e.g. 3 days ago).
    # Note: the word 'ago' is required.
    output_text = re.sub(r'\b\d+\s+(day|days|month|year|months|years)\s+ago\b', '', output_text)

    # Remove possible links (e.g. http://amazon.com)
    msg = ' url '
    output_text = re.sub(r'https?://\S+|http?://\S+|ftp://\S+', msg, output_text)  # https://amazon.con
    output_text = re.sub('www.?\S+', msg, output_text)  # www.amazon.com
    # output_text = re.sub(r'\S+.(?:[a-z])?/\S+|\S+\.(?:[a-z])\S+', msg, output_text)  # amazon.fr/endpoint or
    # amazon.com

    # Remove possible email addresses (e.g. john.doe@yahoo.com)
    msg = ' email '
    output_text = re.sub(r'\S+@\S+|@\S+', msg, output_text)

    # Remove possible html code (e.g. <span style={...}></span>)
    output_text = re.sub(r'<[^>]+>', ' ', output_text)

    # Remove unicode words (e.g. \u200ni)
    output_text = re.sub(r'\\u\S+', '', output_text)

    # Replace common words' contractions that matches any of the dictionary keys
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in cont_common.keys()) + r')\b'
    output_text = re.sub(pattern, lambda match: cont_common[match.group(0)], output_text)

    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in cont_abbreviations.keys()) + r')\b'
    funct = lambda match: cont_abbreviations.get(match.group(0).lower(), match.group(0))
    output_text = re.sub(pattern, funct, output_text, flags=re.IGNORECASE)

    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in cont_time.keys()) + r')\b'
    funct = lambda match: cont_time.get(match.group(0).lower(), match.group(0))
    output_text = re.sub(pattern, funct, output_text, flags=re.IGNORECASE)

    # Convert abbreviated quantities, timing and money symbols to normal words
    output_text = convert_quantities(output_text)
    output_text = convert_time(output_text)
    output_text = convert_money_symbols(output_text)

    # Remove special characters
    for char in ["☆", "★", "♡", "♪"]:
        output_text = re.sub(re.escape(char), '', output_text)

    # Remove words with repeated characters (e.g. happyzzzz, soooo)
    output_text = re.sub(r'(\w)\1{2,}', repl, output_text)

    # Remove repetitions of at least two consecutive punctuations.
    # This doesn't affect punctuations like ??, !! and ...
    output_text = re.sub(r'[-_*+\]\[\\/)(:]{2,}', '', output_text)

    # Remove consecutive points (e.g. '..' or '...')
    output_text = re.sub(r'\.{2,}', '', output_text)

    # Remove suffix '-ish'
    output_text = re.sub(r'-ish\b', '', output_text)  # red-ish -> red

    # Remove hyphens preceded by space or at the beginning of a word
    # which is common for bullet points. For example:
    # '-Word' -> 'Word', '- Word' -> 'Word', but 'word-word' remains
    # same.
    output_text = re.sub(r'-\s-', ' ', output_text)
    output_text = re.sub(r'\s-(?=\w)', ' ', output_text)
    output_text = re.sub(r'(?<!\w)-\s', '', output_text)

    # Convert string with characters separated by punctuations,
    # e.g. w.w.w -> www, w.w.w. -> www.
    output_text = re.sub(r'\.(\w+)', r'\1', output_text)

    # Convert words associated with underslash
    # this_is_great -> this is great
    output_text = re.sub(r'\b(?<=\w)_+(?=\w)\b', ' ', output_text)
    output_text = re.sub(r'\b(?<=\w)/+(?=\w)\b', ' ', output_text)

    # Remove specific punctuations
    puncts = '"#%\'()*+/:,;<=>@[\\]^`{|}~'
    output_text = re.sub('[%s]' % re.escape(puncts), '', output_text)

    # Remove repetitions of 2 or more 'ha' word (e.g. 'hahaha', 'ha ha ha')
    output_text = re.sub(r'\b(ha(?:\s*ha)+)\b', '', output_text)
    output_text = re.sub(r'\b(a(?:\s*ha)+)\b', '', output_text)
    output_text = re.sub(r'\b(A(?:\s*HA)+)\b', '', output_text)
    output_text = re.sub(r'\b(mwaha(?:\s*ha)+)\b', '', output_text)
    output_text = re.sub(r'\b(MWAHA(?:\s*HA)+)\b', '', output_text)
    output_text = re.sub(r'\b(Mwaha(?:\s*ha)+)\b', '', output_text)
    output_text = re.sub(r'\b(Ha(?:\s*Ha)+)\b', '', output_text)
    output_text = re.sub(r'\b(Ha(?:\s*ha)+)\b', '', output_text)
    output_text = re.sub(r'\b(HA(?:\s*HA)+)\b', '', output_text)

    return output_text


def convert_quantities(text: str) -> str:
    """Returns text whose quantity names are converted."""
    # Convert volumes
    output_text = re.sub(r'\b(\d+)\s*ml\b', r'\1 milliliters', text)  # 30ml or 30 ml -> 30 millimeters
    output_text = re.sub(r'\b(\d+)\s*mls\b', r'\1 milliliters', output_text)  # 30ml or 30 ml -> 30 millimeters
    output_text = re.sub(r'\b(\d+)\s*cl\b', r'\1 centiliters', output_text)  # 30ml or 30 ml -> 30 millimeters
    output_text = re.sub(r'\b(\d+)\s*cls\b', r'\1 centiliters', output_text)  # 30ml or 30 ml -> 30 millimeters

    output_text = re.sub(r'\b(\d+)ml-(\d+)ml\b', r'\1 to \2 milliliters', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)ml\b', r'\1 to \2 milliliters', output_text)
    output_text = re.sub(r'\b(\d+)mls-(\d+)mls\b', r'\1 to \2 milliliters', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)mls\b', r'\1 to \2 milliliters', output_text)

    output_text = re.sub(r'\b(\d+)cl-(\d+)cl\b', r'\1 to \2 centiliters', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)cl\b', r'\1 to \2 centiliters', output_text)
    output_text = re.sub(r'\b(\d+)cls-(\d+)cls\b', r'\1 to \2 centiliters', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)cls\b', r'\1 to \2 centiliters', output_text)

    return output_text


def convert_time(text: str) -> str:
    # Convert time
    output_text = re.sub(r'\b(\d+)h\b', r'\1 hours', text)  # 2h -> 2 hours
    output_text = re.sub(r'\b(\d+)H\b', r'\1 hours', output_text)  # 2h -> 2 hours
    output_text = re.sub(r'\b(\d+)hr\b', r'\1 hours', output_text)  # 2h -> 2 hours
    output_text = re.sub(r'\b(\d+)HR\b', r'\1 hours', output_text)  # 2h -> 2 hours
    output_text = re.sub(r'\b(\d+)hrs\b', r'\1 hours', output_text)  # 2h -> 2 hours
    output_text = re.sub(r'\b(\d+)HRS\b', r'\1 hours', output_text)  # 2h -> 2 hours
    output_text = re.sub(r'\b(\d+)\s*sec\b', r'\1 seconds', output_text)
    output_text = re.sub(r'\b(\d+)\s*SEC\b', r'\1 seconds', output_text)
    output_text = re.sub(r'\b(\d+)\s*min\b', r'\1 minutes', output_text)
    output_text = re.sub(r'\b(\d+)\s*MIN\b', r'\1 minutes', output_text)
    output_text = re.sub(r'\b(\d+)\s*mins\b', r'\1 minutes', output_text)
    output_text = re.sub(r'\b(\d+)\s*MINS\b', r'\1 minutes', output_text)

    output_text = re.sub(r'\b(\d+)h-(\d+)h\b', r'\1 to \2 hours', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)h\b', r'\1 to \2 hours', output_text)
    output_text = re.sub(r'\b(\d+)hrs-(\d+)hrs\b', r'\1 to \2 hours', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)hrs\b', r'\1 to \2 hours', output_text)

    output_text = re.sub(r'\b(\d+)min-(\d+)h\b', r'\1 to \2 minutes', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)min\b', r'\1 to \2 minutes', output_text)
    output_text = re.sub(r'\b(\d+)mins-(\d+)mins\b', r'\1 to \2 minutes', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)mins\b', r'\1 to \2 minutes', output_text)

    output_text = re.sub(r'\b(\d+)sec-(\d+)h\b', r'\1 to \2 seconds', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)sec\b', r'\1 to \2 seconds', output_text)

    # Convert date (yr, sec, month, days...)
    output_text = re.sub(r'\b(\d+)\s*yr\b', r'\1 years', output_text)
    output_text = re.sub(r'\b(\d+)\s*yrs\b', r'\1 years', output_text)
    output_text = re.sub(r'\b(\d+)\s*YR\b', r'\1 years', output_text)
    output_text = re.sub(r'\b(\d+)\s*YRS\b', r'\1 years', output_text)
    output_text = re.sub(r'\b(\d+)day\b', r'\1 days', output_text)
    output_text = re.sub(r'\b(\d+)days\b', r'\1 days', output_text)
    output_text = re.sub(r'\b(\d+)DAY\b', r'\1 days', output_text)
    output_text = re.sub(r'\b(\d+)DAYS\b', r'\1 days', output_text)
    output_text = re.sub(r'\b(\d+)month\b', r'\1 months', output_text)
    output_text = re.sub(r'\b(\d+)MONTH\b', r'\1 months', output_text)
    output_text = re.sub(r'\b(\d+)months\b', r'\1 months', output_text)
    output_text = re.sub(r'\b(\d+)MONTHS\b', r'\1 months', output_text)

    output_text = re.sub(r'\b(\d+)-(\d+)day\b', r'\1 to \2 days', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)days\b', r'\1 to \2 days', output_text)
    output_text = re.sub(r'\b(\d+)days-(\d+)days\b', r'\1 to \2 days', output_text)
    output_text = re.sub(r'\b(\d+)day-(\d+)day\b', r'\1 to \2 days', output_text)

    output_text = re.sub(r'\b(\d+)-(\d+)month\b', r'\1 to \2 month', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)months\b', r'\1 to \2 month', output_text)
    output_text = re.sub(r'\b(\d+)months-(\d+)months\b', r'\1 to \2 month', output_text)
    output_text = re.sub(r'\b(\d+)month-(\d+)month\b', r'\1 to \2 month', output_text)

    output_text = re.sub(r'\b(\d+)-(\d+)yr\b', r'\1 to \2 years', output_text)
    output_text = re.sub(r'\b(\d+)-(\d+)yrs\b', r'\1 to \2 years', output_text)
    output_text = re.sub(r'\b(\d+)yr-(\d+)yr\b', r'\1 to \2 years', output_text)
    output_text = re.sub(r'\b(\d+)yrs-(\d+)yrs\b', r'\1 to \2 years', output_text)

    # Convert time indicator (AM, PM)
    output_text = re.sub(r'\b(\d+)\s*am\b', r'\1 before midday', output_text)
    output_text = re.sub(r'\b(\d+)\s*AM\b', r'\1 before midday', output_text)
    output_text = re.sub(r'\b(\d+)\s*pm\b', r'\1 after midday', output_text)
    output_text = re.sub(r'\b(\d+)\s*PM\b', r'\1 after midday', output_text)

    return output_text


def convert_money_symbols(text: str) -> str:
    # Convert monetary strings
    symbols = ['$', '€', '£', '¥', '₹', '₽']
    names = ['dollars', 'euros', 'pound sterlings', 'yens', 'rupees', 'ruoubles']
    output_text = text
    for symb, name in zip(symbols, names):
        pattern = fr'(?:\{symb}(\d+(?:\.\d+)?))|(\d+(?:\.\d+)?)\s*\{symb}'  # 5.99$, $4 -> 5.99 dollars 4 dollars
        output_text = re.sub(pattern, fr'\1\2 {name}', output_text)

    return output_text


def check_cond(a: str, b: str) -> bool:
    """Returns True if a short word is part of a longer word."""
    a = a.lower().strip()
    b = b.lower().strip()

    if len(a) > len(b):
        inner, outer = a, b
    else:
        outer, inner = a, b

    return outer in inner


def is_similar(a: str, b: str) -> bool:
    return a.lower().strip() == b.lower().strip()


def rule_out_extra_ending(text: str, end_interval: int = 20) -> str:
    """Returns input text for which the end part is ruled out if
    the name of target language for translation appears. This extra
    language word is sometimes added by the MarianMT translator.
    As such, we filter it out."""
    output_text = text
    for lang in language_corpus:
        if lang.lower() in text[-end_interval:].lower():  # checks the N last characters of text
            pattern = f'\s?({lang}|{lang.capitalize()})\.$|\s?({lang}|{lang.capitalize()})$'
            output_text = re.sub(pattern, '', text)
            break
    return output_text


def rule_out_by_size(text: str, minimum_length: int = 15) -> Optional[str]:
    """Returns input text if its length exceed given minimum."""
    if len(text) >= minimum_length:
        return text
    else:
        return None
