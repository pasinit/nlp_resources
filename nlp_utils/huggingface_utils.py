from typing import Tuple, List, Dict

import numpy as np
from transformers import PreTrainedTokenizer
import torch

def get_tokenizer_kwargs(model_name):
    if model_name.startswith("gpt2") or model_name.startswith("roberta"):
        return {"add_prefix_space": True}
    return {}


def get_needed_start_end_sentence_tokens(model_name, tokeniser: PreTrainedTokenizer):
    if model_name.startswith("bert") or model_name.startswith("distlbert"):
        return tokeniser.cls_token, tokeniser.sep_token
    elif model_name.startswith("roberta"):
        return tokeniser.bos_token, tokeniser.eos_token
    else:
        return None, None


def encode_word_pieces(tokeniser: PreTrainedTokenizer, sentences: np.ndarray, token_limit, model_name) -> \
        Tuple[List[List[str]], List[List[int]], List[List[int]], List[List[bool]], List[List[List[int]]]]:
    all_tok2seg = list()
    all_segment_str: List[List[str]] = list()
    all_segment_ids: List[List[int]] = list()
    attention_mask: List[List[bool]] = list()
    all_segment_types: List[List[int]] = list()
    all_labels: List[List[int]] = list()
    for s_i in range(len(sentences)):
        sent_tokens = sentences[s_i]
        i_labels = [None] * len(sent_tokens)
        tok2seg = list()
        segs: List[str] = list()
        loc_attention = list()
        segment_types = list()
        loc_labels = list()
        curr_id = 0
        seg_counter = 0
        for tok, label in zip(sent_tokens[:token_limit if token_limit > 0 else len(sent_tokens)], i_labels):
            segments = tokeniser.tokenize(tok, **get_tokenizer_kwargs(model_name))
            mask = [True] * len(segments)
            start_idx_seg = len(segs)
            segs.extend(segments)
            loc_attention.extend(mask)
            segment_types.extend([curr_id] * len(segments))
            tok2seg.append(list(range(start_idx_seg, start_idx_seg + len(segments))))
            seg_counter += len(segments)
        all_segment_ids.append(tokeniser.encode(segs))
        all_tok2seg.append(tok2seg)
        all_segment_str.append(segs)
        attention_mask.append(loc_attention)
        all_segment_types.append(segment_types)

    return all_segment_str, all_segment_ids, all_segment_types, attention_mask, all_tok2seg


def get_model_kwargs(model_name, device, kwargs, type_ids, mask):
    # if model_name.startswith("distilbert"):
    #     mask = torch.LongTensor(mask).to(device) if mask is not None else None
    #     if mask is not None:
    #         if len(mask.shape) < 2:
    #             mask = mask.unsqueeze(0)
    #     kwargs["attention_mask"] = mask
    #     return kwargs

    token_type_ids = type_ids if type_ids is not None else None
    if token_type_ids is not None:
        if len(token_type_ids.shape) < 2:
            token_type_ids = token_type_ids.unsqueeze(0)
    mask = mask if mask is not None else None
    if mask is not None:
        if len(mask.shape) < 2:
            mask = mask.unsqueeze(0)

    kwargs["token_type_ids"] = token_type_ids
    kwargs["attention_mask"] = mask
    return kwargs
