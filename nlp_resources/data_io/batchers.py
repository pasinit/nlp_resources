import math
from abc import ABC, abstractmethod
from random import shuffle

import torch
from torch.utils.data.dataset import IterableDataset
from transformers import PreTrainedTokenizer
from typing import Iterator

import numpy as np

from nlp_resources.nlp_utils.huggingface_utils import get_needed_start_end_sentence_tokens


def __get_batched_elem(token_limit, elem, prefix, postfix, pad):
    if elem is None:
        return None
    if prefix is not None:
        elem = [prefix] + elem
    if postfix is not None:
        elem = elem + [postfix]
    elem = elem + [pad] * (token_limit - len(elem))
    return elem


def __clean_lists(seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx):
    del seg_batch
    del type_ids_batch
    del mask_batch
    del segidx2batchidx
    if tok2seg_batch is not None:
        del tok2seg_batch
        tok2seg_batch = list()
    seg_batch = list()
    type_ids_batch = list()
    mask_batch = list()
    segidx2batchidx = list()
    return seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx


def __pad_and_add(model_name, token_limit, ids, types, mask, tok2seg, seg_batch, type_ids_batch, mask_batch,
                  tok2seg_batch, tokeniser):
    start, end = get_needed_start_end_sentence_tokens(model_name, tokeniser)
    pad_value = tokeniser.pad_token_id if "pad_token" in tokeniser.special_tokens_map is not None else tokeniser.eos_token_id if "eos_token_id" in tokeniser.special_tokens_map else tokeniser.unk_token_id
    ids = __get_batched_elem(token_limit, ids, tokeniser.encode(start, add_special_tokens=False)[0] if start else None,
                             tokeniser.encode(end, add_special_tokens=False)[0] if end else None, pad_value)
    types = __get_batched_elem(token_limit, types, 0 if start else None, 0 if end else None, 0)
    mask = __get_batched_elem(token_limit, mask, 1 if start else None, 1 if end else None, 0)
    if tok2seg is not None:
        tok2seg = __get_batched_elem(token_limit, tok2seg, [] if start else None, [] if end else None, [])

    seg_batch.append(ids)
    type_ids_batch.append(types)
    mask_batch.append(mask)
    if tok2seg_batch is not None:
        tok2seg_batch.append(tok2seg)


def __get_data_to_yield(seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx, device="cuda"):
    if len(segidx2batchidx[-1]) == 0:
        segidx2batchidx = segidx2batchidx[:-1]
    return {"seg": torch.LongTensor(seg_batch).to(device),
            "type": torch.LongTensor(type_ids_batch).to(device) if not all(x is None for x in type_ids_batch) else None,
            "mask": torch.LongTensor(mask_batch).to(device) if not all(x is None for x in mask_batch) else None,
            "tok2seg": tok2seg_batch, "segid2batchidx": segidx2batchidx}


def get_batcher(model_name, all_segments, token_type_ids, attention_mask, tokeniser: PreTrainedTokenizer,
                token_limit,
                device,
                tok2seg=None,
                batch_size=None):
    """
    define an iterator that yields a batched version of the input segments, type_ids and masks and returns it.
    :param model_name: str,
    :param all_segments: List[List[int]], list of lists of token ids.
    :param token_type_ids: List[List[int]], list of lists of token types.
    :param attention_mask: List[List[int]], list of lists of masking.
    :param batch_size: int, size of each batch to yield.
    :param add_cls: bool, it indicates whether to add or not the CLS token at the beginning of each batch elem.
    :param add_sep: bool, it indicates whether to add or not the SEP token at the end of each batch elem.
    :return: an iterator over the batches of all_segments, token_type_ids and attention_mask
    """
    if batch_size is None:
        batch_size = 32

    non_starting_segments2 = list()
    for seq_segments in all_segments:
        non_starting_segments2.append(set(seq_segments[1:]))
    seg2token = None
    non_starting_segments = list()
    if tok2seg is not None:
        seg2token = list()
        for i, token_seg_idxs in enumerate(tok2seg):
            non_starting_segments.append(set())
            seg2token.append(dict())
            for j, seg_idxs in enumerate(token_seg_idxs):
                if len(seg_idxs) > 1:
                    non_starting_segments[-1].update(seg_idxs[1:])
                for idx in seg_idxs:
                    seg2token[-1][idx] = j

    def batch_generator():
        """
        :return: yields a batch in the form of a dict containing:
         - seg: a Tensor with shape [batch_size, max_len] with the token ids padded to the maximum length.
         - type a Tensor with shape [batch_size, max_len] with the token types padded to the maximum length.
         - mask a Tensor with shape [batch_size, max_len] with the masking values padded to the maximum length.
         - tok2seg a List where each elem i is another List where each elem j is the list of indexes in seg that
          correspond to the token j.
         - segid2batchidx a List where each elem i is another List containing the indexes in the batch corresponding
         to the i-th sentence in the input list of segments (i.e., all_segments in the parent function).
        """
        seg_batch = list()
        type_ids_batch = list()
        mask_batch = list()
        tok2seg_batch = None
        if seg2token is not None:
            tok2seg_batch = list()
        segidx2batchidx = list()
        for i in range(len(all_segments)):
            i_non_starting_segments, i_seg2tok, ids, mask, t2s, types = get_ith_elements(i, all_segments,
                                                                                         token_type_ids, attention_mask,
                                                                                         tok2seg, non_starting_segments,
                                                                                         seg2token)
            if len(
                    ids) < token_limit - 2:  ## if this sentence does not exceed token limit just pad and add to the batch
                __pad_and_add(model_name, token_limit, ids, types, mask, t2s, seg_batch, type_ids_batch, mask_batch,
                              tok2seg_batch,
                              tokeniser)
                segidx2batchidx.append([len(seg_batch) - 1])

            else:  # otherwhise, split the sentence into multiple entries of the current batch, pad, add cls, sep each of them separatelly
                mask_batch, seg_batch, segidx2batchidx, tok2seg_batch, type_ids_batch = yield from split_batch_and_yield(
                    i_non_starting_segments, i_seg2tok, ids, mask, mask_batch, seg_batch, segidx2batchidx, t2s,
                    tok2seg_batch, type_ids_batch, types, device)

            ## if we reahed the maximum size of the batch yield the batch
            if len(seg_batch) == batch_size:
                yield __get_data_to_yield(seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx,
                                          device=device)
                seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx = __clean_lists(
                    seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)
        ## if something has been left in the last batch then yield it
        if len(seg_batch) > 0:
            yield __get_data_to_yield(seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx,
                                      device=device)
            __clean_lists(seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)

    def split_batch_and_yield(i_non_starting_segments, i_seg2tok, ids, mask, mask_batch, seg_batch, segidx2batchidx,
                              t2s, tok2seg_batch, type_ids_batch, types, device):
        segidx2batchidx.append([])
        end_index = 0
        k = 0
        # k_limit = max(i_seg2tok.values())
        if len(seg_batch) + math.ceil(len(ids) / (token_limit - 2)) > batch_size:
            ## if I have to split a sentence into two different batches then yield this batch and start
            ## a new one
            yield __get_data_to_yield(seg_batch, type_ids_batch, mask_batch, tok2seg_batch,
                                      segidx2batchidx, device)
            seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx = __clean_lists(
                seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)

        while k < len(
                ids):  ## while We haven't covered all the segments of the current sentence (not considering CLS e SEP)
            end_index = get_end_index(end_index, i_non_starting_segments, ids, k)
            if k == end_index:
                ## the next word is divided in a number of segments that is larger than the maximum number
                ## of segments that can be fit in a batch element. We therefore raise an exception as we do not want to
                ## split words across multiple batch elements.
                raise RuntimeError(
                    "Found a token that start at position {} which is split in too many subtokens (> {}) to be fit"
                    " into a batch element. Check your data or your parameter!\n Here it is the list of segments {}".format(
                        k, token_limit, ids))
            if seg2token is not None:
                end_tok_idx, start_tok_idx = get_start_end_token_index(end_index, i_seg2tok, ids, k, t2s)
            ## pad and add each new entry
            __pad_and_add(model_name, token_limit, ids[k:end_index], types[k:end_index] if types else None,
                          mask[k:end_index] if mask else None,
                          t2s[start_tok_idx:end_tok_idx] if seg2token is not None else None,
                          seg_batch, type_ids_batch, mask_batch,
                          tok2seg_batch, tokeniser)
            if len(segidx2batchidx) == 0:
                segidx2batchidx.append([])
            segidx2batchidx[-1].append(len(seg_batch) - 1)
            ## if during the splitting we reached the maximum batch size then yield the batch
            ## and continue to process the sentence.
            if len(seg_batch) == batch_size:
                yield __get_data_to_yield(seg_batch, type_ids_batch, mask_batch, tok2seg_batch,
                                          segidx2batchidx, device)
                seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx = __clean_lists(
                    seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)

            k = end_index
        ## when the sentence has been processed in full then return the last built batch to continue its
        ## construction
        return mask_batch, seg_batch, segidx2batchidx, tok2seg_batch, type_ids_batch

    def get_ith_elements(i, all_segments, token_type_ids, attention_mask, tok2seg, non_starting_segments, seg2token):
        ids = all_segments[i]
        types = token_type_ids[i] if token_type_ids else None
        mask = attention_mask[i] if attention_mask else None
        if tok2seg is not None:
            t2s = tok2seg[i]
            i_non_starting_segments = non_starting_segments[i]
        if seg2token is not None:
            i_seg2tok = seg2token[i]
        return i_non_starting_segments if seg2token else None, i_seg2tok if seg2token else None, ids, mask, t2s if tok2seg else None, types

    def get_start_end_token_index(end_index, i_seg2tok, ids, k, t2s):
        start_tok_idx = i_seg2tok[k]
        end_tok_idx = i_seg2tok[end_index if end_index < len(ids) else end_index - 1]
        # if end_tok_idx == len(t2s) - 1:
        #     end_tok_idx += 1
        return end_tok_idx, start_tok_idx

    def get_end_index(end_index, i_non_starting_segments, ids, k):
        for j in range(min(k + token_limit - 2, len(ids)) if token_limit > 0 else len(ids), 0, -1):
            if (not j in i_non_starting_segments) or (j == len(ids) - 1):
                end_index = j
                break
        return end_index

    return batch_generator


class ResettableIterator(Iterator, ABC):
    @abstractmethod
    def reset(self):
        pass


class ResettableListIterator(ResettableIterator):
    def __init__(self, collection, shuffle=True):
        self.collection = collection
        self.shuffle = shuffle
        self.reset()

    def __next__(self):
        if len(self.idxs) == 0:
            self.reset()
            raise StopIteration()
        return self.collection[self.idxs.pop()]

    def reset(self):
        self.idxs = list(range(len(self.collection)))
        if self.shuffle:
            shuffle(self.idxs)


class TextDataset(IterableDataset):
    def __init__(self, string_stream: ResettableListIterator, model_name, tokeniser, token_limit, device, batch_size,
                 max_sentences_in_memory=10000):
        """
        :param string_stream: Iterator over sentences that have been already tokenised.
        :param model_name: name of the model to feed with this dataset.
        :param tokeniser: tokeniser corresponding to the model.
        :param token_limit: maximum number of token per sentence.
        :param device: where to put the tensors.
        :param batch_size: size of the batch.
        :param max_sentences_in_memory: maximum number of sentences to keep in memory.
        """
        self.string_stream = string_stream
        self.max_sentences_in_memory = max_sentences_in_memory
        self.model_name = model_name
        self.tokeniser = tokeniser
        self.token_limit = token_limit
        self.device = device
        self.batch_size = batch_size
        self.batch_buffer = list()
        self.stop_iteration = False

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.batch_buffer) > 0:
            return self.batch_buffer.pop()
        elif self.stop_iteration:
            self.stop_iteration = False
            self.string_stream.reset()
            raise StopIteration()
        else:
            self.fill_buffer()
            return self.batch_buffer.pop()

    def fill_buffer(self):
        string_batch = list()
        for _ in range(self.max_sentences_in_memory):
            try:
                string_batch.append(next(self.string_stream))
            except StopIteration:
                self.stop_iteration = True
        all_segments_str, all_segments, token_type_ids, attention_mask, all_tok2seg = encode_word_pieces(
            self.tokeniser, np.array(string_batch), self.token_limit, self.model_name)
        batch_iterator = get_batcher(self.model_name, all_segments, token_type_ids, attention_mask,
                                     self.tokeniser,
                                     self.token_limit, self.device, tok2seg=all_tok2seg,
                                     batch_size=self.batch_size)
        for batch in batch_iterator():
            segments, type_ids, mask, tok2seg, oldidx2newidx = [batch[x] for x in
                                                                ["seg", "type", "mask", "tok2seg",
                                                                 "segid2batchidx"]]
            self.batch_buffer.append((segments, type_ids, mask, tok2seg, oldidx2newidx))
