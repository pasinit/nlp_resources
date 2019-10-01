from enum import Enum
from typing import List, Tuple

from numpy.compat import contextlib_nullcontext
from torch.nn import Module
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction
from deprecated import deprecated


class BertNames(Enum):
    BERT_BASE_UNCASED = "bert-base-uncased"
    BERT_BASE_CASED = "bert-base-cased"
    BERT_BASE_MULTILINGUAL_UNCASED = "bert-base-multilingual-uncased"
    BERT_BASE_MULTILINGUAL_CASED = "bert-base-multilingual-cased"
    BERT_BASE_CHINESE = "bert-base-chinese"
    BERT_BASE_GERMAN_CASED = "bert-base-german-cased"
    BERT_LARGE_UNCASED = "bert-large-uncased"
    BERT_LARGE_UNCASED_WHOLE_WORD_MASKING = "bert-large-uncased-whole-word-masking"
    BERT_LARGE_CASED_WHOLE_WORD_MASKING = "bert-large-cased-whole-word-masking"
    BERT_LARGE_UNCASED_WHOLE_WORD_MASKING_FINETUNED_SQUAD = "bert-large-uncased-whole-word-masking-finetuned-squad"


class BertTokeniserWrapper():
    def __init__(self, bert_model, device, token_limit=100):
        self.token_limit = token_limit
        self.device = device
        self.bert_tokeniser = BertTokenizer.from_pretrained(bert_model)

    def __add_special_tokens(self, sentences):
        new_sentences = list()
        for words in sentences:
            toks_a = words
            toks_b = None
            if type(words) == tuple:
                toks_a, toks_b = list(zip(*words))
            words = "[CLS] " + toks_a + " [SEP]"
            if toks_b is not None:
                words + " " + toks_b + " [SEP]"
            tokens = self.bert_tokeniser.tokenize(words)
            new_sentences.append(tokens)
        return new_sentences

    def merge_words(self, words):
        merged = list()
        s_mapping = list()
        i = 0
        for j, w in enumerate(words):
            # s_mapping.append(i)
            if w.startswith("##"):  ## avoid to merge words that had ## at the beginning not because the segmentation.
                merged[-1] += w.replace("##", "")
                s_mapping.append(s_mapping[-1])
            else:
                merged.append(w)
                s_mapping.append(s_mapping[-1] + 1 if len(s_mapping) > 0 else 0)

            # if j < len(words) - 1 and not words[j+1].startswith("##"):
            #     i += 1
        return merged, s_mapping

    def tokenise(self, sentences):
        indexed_tokens = list()
        segments_class = list()
        attention_masks = list()
        new_sentences = list()
        for words in sentences:
            toks_a = words
            toks_b = None
            if type(words) == tuple:
                toks_a, toks_b = words
            new_words = "[CLS] " + toks_a + " [SEP]" + (" " + toks_b + " [SEP]" if toks_b is not None else "")
            tokens_all = self.bert_tokeniser.tokenize(new_words)
            new_sentences.append(tokens_all)
            ids_all = self.bert_tokeniser.convert_tokens_to_ids(tokens_all)
            mask = [1] * len(ids_all)
            first_sep_idx = tokens_all.index("[SEP]")
            tokens_class = [0] * (first_sep_idx + 1) + (
                [1] * (len(tokens_all) - first_sep_idx - 1) if toks_b is not None else [])

            indexed_tokens.append(ids_all)
            attention_masks.append(mask)
            segments_class.append(tokens_class)

        max_len = min(max([len(s) for s in indexed_tokens]), self.token_limit)
        new_index_tokens = list()
        new_att_masks = list()
        new_tokens_class = list()
        for tokens, segment_ids, attention_mask in zip(indexed_tokens, segments_class, attention_masks):
            if max_len > 0:
                if len(tokens) < max_len:
                    segment_ids = segment_ids + [0] * (max_len - len(tokens))
                    attention_mask = attention_mask + [0] * (max_len - len(tokens))
                    tokens = tokens + ([self.bert_tokeniser.vocab["[PAD]"]] * (max_len - len(tokens)))

                else:
                    tokens = tokens[:max_len]
                    segment_ids = segment_ids[:max_len]
                    attention_mask = attention_mask[:max_len]
            new_att_masks.append(attention_mask)
            new_index_tokens.append(tokens)
            new_tokens_class.append(segment_ids)

        indexed_tokens = torch.LongTensor(new_index_tokens).to(self.device)
        tokens_class = torch.LongTensor(new_tokens_class).to(self.device)
        attention_masks = torch.LongTensor(new_att_masks).to(self.device)
        return new_sentences, indexed_tokens, tokens_class, attention_masks

    def pad_list(self, l, size, pad_val):
        padded_l = list()
        for x in l:
            if len(x) < size:
                x = x + [pad_val] * (size - len(x))
            padded_l.append(x)
        return padded_l

    def segment_tokenised_words(self, sentences: np.ndarray) -> Tuple[
        List[List[str]], List[List[int]], List[List[int]], List[List[bool]], List[List[List[int]]]]:
        all_tok2seg = list()
        all_segment_str = list()
        all_segment_ids = list()
        attention_mask = list()
        all_segment_types = list()
        for sent_tokens in sentences:
            tok2seg = list()
            segs = list()
            loc_attention = list()
            segment_types = list()
            curr_id = 0
            seg_counter = 0
            for tok in sent_tokens[:self.token_limit if self.token_limit > 0 else len(sent_tokens)]:
                segments = self.bert_tokeniser.wordpiece_tokenizer.tokenize(tok)
                mask = [x != self.bert_tokeniser.pad_token for x in segments]
                start_idx_seg = len(segs)
                segs.extend(segments)
                loc_attention.extend(mask)
                segment_types.extend([curr_id] * len(segments))
                tok2seg.append(list(range(start_idx_seg, start_idx_seg + len(segments))))

                if tok == self.bert_tokeniser.sep_token:
                    curr_id = 1
                seg_counter += len(segments)
            all_segment_ids.append(self.bert_tokeniser.convert_tokens_to_ids(segs))
            all_tok2seg.append(tok2seg)
            all_segment_str.append(segs)
            attention_mask.append(loc_attention)
            all_segment_types.append(segment_types)
        # padded_len = max([len(x) for x in all_segment_ids])
        # all_segment_ids = self.pad_list(all_segment_ids, padded_len,
        #                                 self.bert_tokeniser.pad_token_id)
        # attention_mask = self.pad_list(attention_mask, padded_len, 0)
        # all_segment_types = self.pad_list(all_segment_types, padded_len, 0)

        return all_segment_str, all_segment_ids, all_segment_types, attention_mask, all_tok2seg


class MergeMode(Enum):
    AVG = "avg"
    SUM = "sum"
    FIRST = "first"


class GenericBertWrapper(Module):
    def __init__(self, bert_model, model_name, device, eval_mode=True, token_limit=256):
        super().__init__()
        self.bert_tokeniser = BertTokeniserWrapper(model_name, device, token_limit=-1)
        if eval_mode:
            bert_model.eval()
        self.bert_model = bert_model.to(device)

        self.device = device
        self.token_limit = token_limit
        self.eval_mode = eval_mode

    def forward(self, sentences, **kwargs):
        str_tokens, tokens, segment_ids, attention_masks = self.bert_tokeniser.tokenise(sentences)
        # if self.eval_mode:
        #     with torch.no_grad():
        #         last_hidden_states, *_ = self.bert_model(tokens, segment_ids, attention_mask=attention_masks, **kwargs)
        # else:
        batch_size = kwargs.get("batch_size", 32)
        batcher = self.__get_batcher(tokens, segment_ids, attention_masks, batch_size, kwargs.get("add_cls", True),
                                     kwargs.get("add_sep", True))
        with self.get_context():
            for data in batcher():
                segments, type_ids, mask, oldidx2newidx = [data[x] for x in
                                                           ["seg", "type", "mask", "segid2batchidx"]]
                bert_out = self.bert_model(segments, token_type_ids=type_ids, attention_mask=mask, **kwargs)
        return {"out": bert_out,
                "bert_in": {"str_tokens": str_tokens, "ids": tokens, "segment_ids": segment_ids,
                            "attention_mask": attention_masks}}

    def __get_batched_elem(self, elem, cls, sep, pad_val):
        if sep is not None:
            elem = elem + [sep]
        elem = elem + [pad_val] * (self.token_limit - len(elem) - 1)
        if cls is not None:
            elem = [cls] + elem
        return elem

    def __clean_lists(self, seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx):
        del seg_batch
        del type_ids_batch
        del mask_batch
        del tok2seg_batch
        del segidx2batchidx
        seg_batch = list()
        type_ids_batch = list()
        mask_batch = list()
        segidx2batchidx = list()
        tok2seg_batch = list()
        return seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx

    def __pad_and_add(self, ids, types, mask, tok2seg, seg_batch, type_ids_batch, mask_batch, tok2seg_batch, cls_id,
                      add_cls,
                      sep_id, add_sep, pad_id):
        ids = self.__get_batched_elem(ids, cls_id if add_cls else None, sep_id if add_sep else None, pad_id)
        types = self.__get_batched_elem(types, 0 if add_cls else None, 0 if add_sep else None, pad_id)
        mask = self.__get_batched_elem(mask, 0 if add_cls else None, 0 if add_sep else None, pad_id)
        tok2seg = self.__get_batched_elem(tok2seg, [] if add_cls else None, [] if add_sep else None, None)

        seg_batch.append(ids)
        type_ids_batch.append(types)
        mask_batch.append(mask)
        tok2seg_batch.append(tok2seg)
    def __get_data_to_yield(self,seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx, device="cuda"):
        return {"seg": torch.LongTensor(seg_batch).to(device), "type": torch.LongTensor(type_ids_batch).to(device),
         "mask": torch.LongTensor(mask_batch).to(device),
         "tok2seg": tok2seg_batch, "segid2batchidx": segidx2batchidx}

    def __get_batcher(self, all_segments, token_type_ids, attention_mask, tok2seg, batch_size=None,
                      add_cls=True, add_sep=True):
        """
        define an iterator that yields a batched version of the input segments, type_ids and masks and returns it.
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

        non_starting_segments = list()
        seg2token = list()
        for i, token_seg_idxs in enumerate(tok2seg):
            non_starting_segments.append(set())
            seg2token.append(dict())
            for j, seg_idxs in enumerate(token_seg_idxs):
                if len(seg_idxs) > 1:
                    non_starting_segments[-1].update(seg_idxs[1:])
                for idx in seg_idxs:
                    seg2token[-1][idx] = j

        pad_id = self.bert_tokeniser.bert_tokeniser.pad_token_id
        cls_id = self.bert_tokeniser.bert_tokeniser.cls_token_id
        sep_id = self.bert_tokeniser.bert_tokeniser.sep_token_id

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
            tok2seg_batch = list()
            segidx2batchidx = list()
            for i in range(len(all_segments)):
                ids = all_segments[i]
                types = token_type_ids[i]
                mask = attention_mask[i]
                t2s = tok2seg[i]
                i_non_starting_segments = non_starting_segments[i]
                i_seg2tok = seg2token[i]
                if len(ids) < self.token_limit:
                    self.__pad_and_add(ids, types, mask, t2s, seg_batch, type_ids_batch, mask_batch, tok2seg_batch,
                                       cls_id, add_cls,
                                       sep_id, add_sep, pad_id)
                    segidx2batchidx.append([len(seg_batch) - 1])

                else:  # split into multiple entries, pad, add cls, sep
                    segidx2batchidx.append([])
                    end_index = 0
                    k = 0
                    while k < len(ids):
                        for j in range(min(k + self.token_limit - 2, len(ids) - 1), 0, -1):
                            if (not j in i_non_starting_segments) or (j == len(ids) - 1):
                                end_index = j
                                break
                        # if k == end_index or end_index == len(ids) - 1:
                        #     end_index = len(ids)
                        # if end_index - k == 0:
                        start_tok_idx = i_seg2tok[k]
                        end_tok_idx = i_seg2tok[end_index]
                        if end_tok_idx == len(t2s) - 1:
                            end_tok_idx += 1
                        if end_index == len(ids) - 1:
                            end_index += 1
                        self.__pad_and_add(ids[k:end_index], types[k:end_index], mask[k:end_index], t2s[start_tok_idx:end_tok_idx],
                                           seg_batch, type_ids_batch, mask_batch,
                                           tok2seg_batch, cls_id, add_cls, sep_id, add_sep, pad_id)
                        segidx2batchidx[-1].append(len(seg_batch) - 1)
                        if len(seg_batch) == batch_size:
                            yield self.__get_data_to_yield(seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)
                            seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx = self.__clean_lists(
                                seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)
                        k = end_index

                if len(seg_batch) == batch_size:
                    yield self.__get_data_to_yield(seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)
                    seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx = self.__clean_lists(
                        seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)
            yield self.__get_data_to_yield(seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)
            self.__clean_lists(seg_batch, type_ids_batch, mask_batch, tok2seg_batch, segidx2batchidx)
        return batch_generator

    def __merge_batch_back(self, batch, batched_tok2seg, oldidx2newidx):
        batch_merged = list()
        tok2seg = list()
        for i in range(len(oldidx2newidx)):
            newidxs = oldidx2newidx[i]
            to_be_merged = batch[newidxs]
            tok2seg_to_be_merged = [batched_tok2seg[x] for x in newidxs]
            seg_ids = [[y for y in x[1:] if y is not None and y !=[]] for x in tok2seg_to_be_merged]
            tok2seg.append(sum(seg_ids, []))
            u_to_be_merged = to_be_merged.unbind()
            new_to_be_merged = list()
            for j in range(len(seg_ids)):
                j_seg_ids = list(range(1, len([x + 1 for y in seg_ids[j] for x in y if x is not None and x != []])+ 1))
                j_u_to_be_merged = u_to_be_merged[j][j_seg_ids]
                new_to_be_merged.append(j_u_to_be_merged)

            merged = torch.cat(new_to_be_merged, 0)
            batch_merged.append(merged)
        assert len(batch_merged) == len(oldidx2newidx)
        return batch_merged, tok2seg


    def word_forward(self, sentences: np.ndarray, **kwargs):
        """
        :param sentences: list of already tokenised sentences, i.e., List[List[str]]
        :param kwargs: optional arguments to pass to bert model
        :return: Tensor with shape [sentences.shape[0], sentences.shape[1], hidden_size]
        """
        if "merge_mode" in kwargs:
            merge_mode = kwargs["merge_mode"]
        else:
            merge_mode = MergeMode.AVG
        all_segments_str, all_segments, token_type_ids, attention_mask, all_tok2seg = self.bert_tokeniser.segment_tokenised_words(
            sentences)
        batch_iterator = self.__get_batcher(all_segments, token_type_ids, attention_mask, all_tok2seg,
                                            kwargs.get("batch_size", None))
        all_bert_out = list()
        with self.get_context():
            for data in batch_iterator():
                segments, type_ids, mask, tok2seg, oldidx2newidx = [data[x] for x in
                                                                    ["seg", "type", "mask", "tok2seg",
                                                                     "segid2batchidx"]]
                bert_out = self.bert_model(segments, token_type_ids=type_ids,
                                           attention_mask=mask, **kwargs)
                last_hidden_states = bert_out[0]
                merged_batch, tok2seg = self.__merge_batch_back(last_hidden_states, tok2seg, oldidx2newidx)
                merged_hidden_states = self.__merge_hidden_states(merged_batch, tok2seg, merge_mode)
                for entry in zip(merged_hidden_states.unbind(), *[x.unbind() for x in bert_out[1:]]):
                    all_bert_out.append(entry)

                # all_bert_out.append([merged_hidden_states] + list(bert_out[1:]))

        return {"out": all_bert_out,
                "bert_in": {"str_tokens": sentences, "ids": all_segments, "token_type_ids": token_type_ids,
                            "attention_mask": attention_mask}}

    def get_context(self):
        if self.eval_mode:
            return torch.no_grad()
        else:
            return contextlib_nullcontext

    def __merge_hidden_states(self, hidden_states, tok2seg: List[List[List[int]]], merge_mode: MergeMode):
        max_size = max([len(x) for x in tok2seg])
        merged_hidden_states = torch.zeros(len(hidden_states), max_size, len(hidden_states[0][0]))

        for i in range(len(hidden_states)):
            hs_i = hidden_states[i]
            t2s_i: List[List[int]] = tok2seg[i]
            merged_hs_i = list()
            for seg_ids in t2s_i:
                hidden_segs = hs_i[np.array(seg_ids)]
                if merge_mode == MergeMode.AVG:
                    merged = torch.mean(hidden_segs, 0)
                elif merge_mode == MergeMode.SUM:
                    merged = torch.sum(hidden_segs, 0)
                else:
                    merged = hidden_segs[0]
                merged_hs_i.append(merged)
            merged_hidden_states[i, :len(merged_hs_i), :] = torch.stack(merged_hs_i, 0)
        return merged_hidden_states

    @deprecated(version='1.0', reason="use word_forward function instead")
    def get_word_hidden_states(self, hidden_states, mapping):
        max_val = 0
        if len(hidden_states.shape) < 3:  # no batch
            hidden_states = hidden_states.unsqueeze(0)
            mapping = [mapping]

        states_counter = torch.zeros(len(mapping), max([x[-1] + 1 for x in mapping])).to(hidden_states.device)

        new_states = torch.zeros(len(mapping), max([x[-1] + 1 for x in mapping]), hidden_states.shape[-1]).to(
            hidden_states.device)

        for i, b_m in enumerate(mapping):
            for j, v in enumerate(b_m):
                if v > max_val:
                    max_val = v
                new_states[i][v] = new_states[i][v] + hidden_states[i][j]
                states_counter[i][v] = states_counter[i][v] + 1
        new_states = new_states / states_counter.unsqueeze(-1)
        return new_states


class BertWrapper(GenericBertWrapper):
    def __init__(self, model_name, device, eval_mode=True, token_limit=100):
        model = BertModel.from_pretrained(model_name)
        super().__init__(model, model_name, device, eval_mode, token_limit)

    def __merge_back_batched_sentences(self, bert_out, pooled_out_merging: MergeMode = None):
        if pooled_out_merging is None:
            pooled_out_merging = MergeMode.AVG
        merged_hidden_states = list()
        merged_pooled_output = list()
        for batch_bert_out, mapping in bert_out["out"]:
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
        bert_out = super(BertWrapper, self).forward(sentences, **kwargs)
        hidden_states, pooled_output = bert_out["out"]
        return {"hidden_states": hidden_states, "sentence_embedding": pooled_output}, \
               bert_out["bert_in"]

    def word_forward(self, sentences: np.ndarray, **kwargs):
        bert_out = super(BertWrapper, self).word_forward(sentences, **kwargs)
        hidden_states, pooled_output = zip(*bert_out["out"])
        # hidden_states, pooled_output, *_ = bert_out["out"]
        return {"hidden_states": torch.stack(hidden_states, 0), "sentence_embedding": torch.stack(pooled_output)}, \
               bert_out["bert_in"]


class BertSentencePredictionWrapper(GenericBertWrapper):
    def __init__(self, model_name, device, token_limit=100):
        model = BertForNextSentencePrediction.from_pretrained(model_name)
        super().__init__(model, model_name, device, True, token_limit)

    def forward(self, sentences, **kwargs):
        if "next_sentence_label" not in kwargs:
            kwargs["next_sentence_label"] = None
        bert_out = super(BertSentencePredictionWrapper, self).forward(sentences, **kwargs)
        sent_class_logits = bert_out["out"]
        return sent_class_logits
