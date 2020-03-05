from enum import Enum

import torch
from numpy.compat import contextlib_nullcontext
from torch.nn import Module
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, AutoConfig
from typing import List

from data_io.batchers import get_batcher
from nlp_models.bert_wrappers import MergeMode
import numpy as np

from nlp_utils.huggingface_utils import encode_word_pieces

from nlp_utils.huggingface_utils import get_model_kwargs


class HuggingfaceModelNames(Enum):
    BERT_LARGE_CASED = "bert-large-cased"
    XLM_ROBERTA_LARGE = "xlm-roberta-large"
    # XLM_MLM_100_1280 = "xlm-mlm-100-1280"
    ROBERTA_BASE = "roberta-base"
    XLNET_BASE_CASED = "xlnet-base-cased"
    OPEN_AI_GPT2_BASE = "gpt2"
    OPEN_AI_GPT2_MEDIUM = "gpt2-medium"
    BERT_BASE_UNCASED = "bert-base-uncased"
    # OPEN_AI_GPT2_large = "gpt2-large"
    # XLNET_LARGE_CASED = "xlnet-large-cased"
    ROBERTA_LARGE = "roberta-large"
    BERT_BASE_CASED = "bert-base-cased"
    BERT_BASE_MULTILINGUAL_UNCASED = "bert-base-multilingual-uncased"
    BERT_BASE_MULTILINGUAL_CASED = "bert-base-multilingual-cased"
    # BERT_BASE_CHINESE = "bert-base-chinese"
    # BERT_BASE_GERMAN_CASED = "bert-base-german-cased"
    BERT_LARGE_UNCASED = "bert-large-uncased"


class GenericHuggingfaceWrapper(Module):
    def __init__(self, model_name, device, eval_mode=True, token_limit=256, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.tokeniser: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        config = AutoConfig.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        if eval_mode:
            self.model.eval()
        self.model.to(device)

        self.device = device
        self.token_limit = token_limit
        self.eval_mode = eval_mode

    def __merge_back_batched_sentences(self, model_out, pooled_out_merging: MergeMode = None):
        if pooled_out_merging is None:
            pooled_out_merging = MergeMode.AVG
        merged_hidden_states = list()
        merged_pooled_output = list()
        for batch_bert_out, mapping in model_out["out"]:
            hidden_states, pooled_output = batch_bert_out
            for indices in mapping:
                indexed_hs = hidden_states[indices[0]][0:-1]
                for i in indices[1:]:
                    if i < len(indices) - 2:
                        indexed_hs.append(
                            hidden_states[i][1:-1])  ## remove cls and sep from the states of each entry in the batch
                    else:
                        indexed_hs.append(hidden_states[i][1:])  ## remove only cls from the entry of the last index
                merged_hidden_states.append(torch.stack(indexed_hs, 1))
                if pooled_out_merging == MergeMode.AVG:
                    mpo = torch.mean(pooled_output[indices], 0)
                elif pooled_out_merging == MergeMode.SUM:
                    mpo = torch.sum(pooled_output[indices], 0)
                else:
                    mpo = pooled_output[indices[0]]
                merged_pooled_output.append(mpo)
        return merged_hidden_states, merged_pooled_output

    def forward(self, sentences, **kwargs):
        tokenised_str = [self.tokeniser.tokenize(sentence) for sentence in sentences]
        encoded_data = [self.tokeniser.encode_plus(toks) for toks in tokenised_str]
        input_ids, token_type_ids, attention_masks = list(), list(), list()

        for encoded_info_str in encoded_data:
            i_input_ids, i_token_type_ids = [encoded_info_str[x] for x in ["input_ids", "token_type_ids"]]
            input_ids.append(i_input_ids)
            token_type_ids.append(i_token_type_ids)
            attention_masks.append([1] * len(i_input_ids))

        batch_size = kwargs.get("batch_size", 32)
        batcher = get_batcher(self.model_name, input_ids, token_type_ids, attention_masks, self.tokeniser,
                              self.token_limit, self.device, batch_size=batch_size)
        with self.get_context():
            for data in batcher():
                segments, type_ids, mask, oldidx2newidx = [data[x] for x in
                                                           ["seg", "type", "mask", "segid2batchidx"]]
                out = self.model(segments, token_type_ids=type_ids, attention_mask=mask, **kwargs)
        return {"out": out,
                "bert_in": {"str_tokens": tokenised_str, "ids": input_ids, "segment_ids": token_type_ids,
                            "attention_mask": attention_masks}}

    def __merge_batch_back(self, batch, batched_tok2seg, oldidx2newidx):
        batch_merged = list()
        tok2seg = list()
        for i in range(len(oldidx2newidx)):
            newidxs = oldidx2newidx[i]
            to_be_merged = batch[newidxs]
            tok2seg_to_be_merged = [batched_tok2seg[x] for x in newidxs]
            seg_ids = [[y for y in x if y is not None and y != []] for x in tok2seg_to_be_merged]
            tok2seg.append(sum(seg_ids, []))
            u_to_be_merged = to_be_merged.unbind()
            new_to_be_merged = list()
            for j in range(len(seg_ids)):
                j_seg_ids = list(range(1, len([x + 1 for y in seg_ids[j] for x in y if x is not None and x != []]) + 1))
                if j >= len(u_to_be_merged):
                    print("WARNING: {} out of range of u_to_be_merged. Skipping the rest.".format(j))
                    return None, None
                if any(x >= len(u_to_be_merged[j]) for x in j_seg_ids):
                    print("WARNING: out of range in u_to_be_merged. Skipping the rest.")
                    return None, None
                j_u_to_be_merged = u_to_be_merged[j][j_seg_ids]
                new_to_be_merged.append(j_u_to_be_merged)
            merged = torch.cat(new_to_be_merged, 0)
            batch_merged.append(merged)
        assert len(batch_merged) == len(oldidx2newidx)
        return batch_merged, tok2seg

    def aggregate_layers(self, tensor, function_name):
        if function_name == "sum":
            return torch.sum(tensor, 0).squeeze()
        if function_name == "mean":
            return torch.mean(tensor, 0).squeeze()
        else:
            raise RuntimeError("function {} not recognised! Choose between \"sum\" and \"mean\"")

    def word_forward(self, segments, type_ids, mask, tok2seg, oldidx2newidx, merge_mode,
                    aggregate_layers = None, layers_aggregation_function="sum", **kwargs):
        kwargs = get_model_kwargs(self.model_name, self.device, kwargs, type_ids, mask)
        model_out = self.model(segments, **kwargs)
        # token_type_ids=type_ids, attention_mask=mask, **kwargs)
        if aggregate_layers:
            assert len(model_out) > 1
            tensors_to_aggregate = torch.stack([model_out[-1][x] for x in aggregate_layers], 0)
            hidden_states = self.aggregate_layers(tensors_to_aggregate, layers_aggregation_function)
        else: hidden_states = model_out[0]

        last_hidden_states = hidden_states[:, 1:, :]
        merged_batch, tok2seg = self.__merge_batch_back(last_hidden_states, tok2seg, oldidx2newidx)
        return self.__merge_hidden_states(merged_batch, tok2seg, merge_mode)

    def sentences_forward(self, sentences: np.ndarray, print_bar=False, **kwargs):
        """
        :param sentences: list of already tokenised sentences, i.e., List[List[str]]
        :param kwargs: optional arguments to pass to bert model
        :return: Tensor with shape [sentences.shape[0], sentences.shape[1], hidden_size]
        """
        if "merge_mode" in kwargs:
            merge_mode = kwargs["merge_mode"]
        else:
            merge_mode = MergeMode.AVG
        all_segments_str, all_segments, token_type_ids, attention_mask, all_tok2seg = encode_word_pieces(
            self.tokeniser, sentences, self.token_limit, self.model_name)
        batch_iterator = get_batcher(self.model_name, all_segments, token_type_ids, attention_mask,
                                     self.tokeniser, self.token_limit,
                                     self.device,
                                     tok2seg=all_tok2seg,
                                     batch_size=kwargs.get("batch_size", None))
        hidden_states = list()
        with self.get_context():
            iterator = batch_iterator()
            if print_bar:
                iterator = tqdm(iterator)
            for data in iterator:
                segments, type_ids, mask, tok2seg, oldidx2newidx = [data[x] for x in
                                                                    ["seg", "type", "mask", "tok2seg",
                                                                     "segid2batchidx"]]
                merged_hidden_states = self.word_forward(segments, type_ids, mask, tok2seg, oldidx2newidx, merge_mode,
                                                         **kwargs)
                if merged_hidden_states is None:
                    print("WARNING: skipping an entire batch of sentences!!!")
                    return {"hidden_states": None}, \
                           {"str_tokens": sentences, "ids": all_segments, "token_type_ids": token_type_ids,
                            "attention_mask": attention_mask}
                for hs in merged_hidden_states.unbind():
                    hidden_states.append(hs)
                # hidden_states.append([merged_hidden_states] + list(model_out[1:]))
        # {"out": hidden_states,
        #        "bert_in": }
        # hidden_states, pooled_output = zip(*hidden_states)
        parallel_hidden_states = list()
        assert len(sentences) == len(hidden_states)
        for s, h in zip(sentences, hidden_states):
            assert torch.sum(h[len(s):, :]) == 0.0
            h = h[:len(s), :].to(self.device)
            parallel_hidden_states.append(h)
        del hidden_states
        return {"hidden_states": parallel_hidden_states}, \
               {"str_tokens": sentences, "ids": all_segments, "token_type_ids": token_type_ids,
                "attention_mask": attention_mask}

    def get_context(self):
        if self.eval_mode:
            return torch.no_grad()
        else:
            return contextlib_nullcontext

    def __merge_hidden_states(self, hidden_states, tok2seg: List[List[List[int]]], merge_mode: MergeMode):
        if hidden_states is None:
            return None
        max_size = max([len(x) for x in tok2seg])
        merged_hidden_states = torch.zeros(len(hidden_states), max_size, len(hidden_states[0][0]))

        for i in range(len(hidden_states)):
            hs_i = hidden_states[i]
            t2s_i: List[List[int]] = tok2seg[i]
            merged_hs_i = list()
            for seg_ids in t2s_i:
                hidden_segs = hs_i[np.array(seg_ids) - 1]
                if merge_mode == MergeMode.AVG:
                    merged = torch.mean(hidden_segs, 0)
                elif merge_mode == MergeMode.SUM:
                    merged = torch.sum(hidden_segs, 0)
                else:
                    merged = hidden_segs[0]
                merged_hs_i.append(merged)
            merged_hidden_states[i, :len(merged_hs_i), :] = torch.stack(merged_hs_i, 0)
            # merged_hidden_states.append(torch.stack(merged_hs_i, 0))
        return merged_hidden_states
