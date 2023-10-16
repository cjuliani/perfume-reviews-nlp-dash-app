from typing import Dict, Any, List, Callable, Optional
from utils.cleaning import is_similar, check_cond


def process_cleaning_per_language(
        text_dict: Dict[str, List[str]],
        cleaning_function: Callable,
        prefix_msg: Optional[str] = None,
        **kwargs: Any) -> Dict[str, List[str]]:
    """Returns cleaned texts per language."""
    processed = {}
    for j, language in enumerate(text_dict):
        processed[language] = []

        for i, text in enumerate(text_dict[language]):
            # Apply filter to current text
            output_text = cleaning_function(text, **kwargs)

            if output_text:
                processed[language].append(cleaning_function(text))

            # ---
            cnt = f'{i + 1}/{len(text_dict[language])}'
            msg = f'{prefix_msg}, processed: {cnt}' if prefix_msg else f'processed: {cnt}'
            print(msg, end='\r')
            yield msg

    yield processed


def rule_out_duplicates_per_language(
        text_dict: Dict[str, List[str]],
        prefix_msg: Optional[str] = None,) -> Dict[str, List[str]]:
    """Returns cleaned texts per language."""
    processed = {}
    for j, language in enumerate(text_dict):
        processed[language] = []
        sent_num = len(text_dict[language])

        seen = []
        duples = []
        for i, text in enumerate(text_dict[language]):
            # Rule out text if seen
            if text not in seen:
                # Even not seen, the text may exist in one of those
                # already seen (both texts may slightly differ). As
                # such, rule out current text if existing in one of
                # the ones seen
                cond1 = any([is_similar(seq, text) for seq in seen])
                cond2 = any([check_cond(seq, text) for seq in seen])
                if seen and (cond1 or cond2):
                    duples.append(text)
                else:
                    seen.append(text)

            # ---
            cnt = f'{i + 1}/{len(text_dict[language])}'
            msg = f'{prefix_msg}, processed: {cnt}' if prefix_msg else f'processed: {cnt}'
            print(msg, end='\r')
            yield msg

        processed[language] = seen

        msg = (f"{sent_num} reviews originally, {len(seen)} " +
               f"remaining ({sent_num - len(seen)} ruled out)")
        yield msg

    yield processed
