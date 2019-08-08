from enum import Enum

from pytorch_pretrained_bert import BertModel, BertForNextSentencePrediction, BertTokenizer
from torch.nn import Module
import torch


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
            if w.startswith("##") and not w.startswith("###"): ## avoid to merge words that had ## at the beginning not because the segmentation.
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


class GenericBertWrapper(Module):
    def __init__(self, bert_model, model_name, device, eval_mode=True, token_limit=100):
        super().__init__()
        self.bert_tokeniser = BertTokeniserWrapper(model_name, device, token_limit=token_limit)
        if eval_mode:
            bert_model.eval()
        self.bert_model = bert_model.to(device)

        self.device = device
        self.token_limit = token_limit
        self.eval_mode = eval_mode

    def forward(self, sentences, **kwargs):
        str_tokens, tokens, segment_ids, attention_masks = self.bert_tokeniser.tokenise(sentences)
        print(tokens.shape)
        if self.eval_mode:
            with torch.no_grad():
                out = self.bert_model(tokens, segment_ids, attention_mask=attention_masks, **kwargs)
        else:
            out = self.bert_model(tokens, segment_ids, attention_mask=attention_masks, **kwargs)
        return {"out": out, "bert_in": {"str_tokens": str_tokens, "ids": tokens, "segment_ids": segment_ids,
                                        "attention_mask": attention_masks}}

    ### TODO test it!
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
    def __init__(self, model_name, device, token_limit=100):
        model = BertModel.from_pretrained(model_name)
        super().__init__(model, model_name, device, True, token_limit)

    def forward(self, sentences, **kwargs):
        if "output_all_encoded_layers" not in kwargs:
            kwargs["output_all_encoded_layers"] = False
        bert_out = super(BertWrapper, self).forward(sentences, **kwargs)
        hidden_states, pooled_output = bert_out["out"]

        # sentence_embeddings = hidden_states[:, 0, :]
        return {"cls_states": hidden_states[:, 0], "hidden_states": hidden_states, "sentence_embedding": pooled_output}, \
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
