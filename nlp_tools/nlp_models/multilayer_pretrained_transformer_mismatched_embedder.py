from typing import Optional, List, Callable, Any

import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util
from overrides import overrides
from torch import Tensor

from nlp_tools.nlp_models.multilayer_pretrained_transformer_embedder import MultilayerPretrainedTransformerEmbedder

# from nlp_tools.nlp_models.multilayer_pretrained_transformer_mismatched_embedder import __word_segment_mean_merger

"""
Most of this code is copied from allennlp library version 1.0 RC4.
"""
def __word_segment_mean_merger(span_embeddings: Tensor, span_mask: Tensor) -> Tensor:
    """
    :param tensor:
    :param span_mask:
    :return:
    """
    span_embeddings_sum = span_embeddings.sum(2)
    span_embeddings_len = span_mask.sum(2)
    # Shape: (batch_size, num_orig_tokens, embedding_size)
    return span_embeddings_sum / span_embeddings_len

def __word_segment_first_merger(span_embeddings: Tensor, *args) -> Tensor:
    """
    :param span_embeddings:
    :param args:
    :return:
    """
    return span_embeddings[:, :, 0, :]

def get_combiner(combiner:str) -> Callable[[Tensor, Any], Tensor]:
    if combiner == "mean":
        return __word_segment_mean_merger
    if combiner == "first":
        return __word_segment_first_merger
    else:
        raise RuntimeError("value {} not recognised for combiner function. Use mean or first")
class MultilayerPretrainedTransformerMismatchedEmbedder(TokenEmbedder):



    def __init__(self, model_name: str, layers_to_merge: List,
                 max_length: int = None,
                 layer_merger: Callable[[List[Tensor]], Tensor] = sum,
                 word_segment_emb_merger: str = "mean"):
        """

        :param model_name: name of the pretrained embedder
        :param layers_to_merge: indices of the layers which output want to be merged into the output embedding
        :param max_length: maximum length of the sequence
        :param layer_merger: function that merges the hidden states of the layers indicated in `layers_to_merge`.
        By default, this is the sum of the layers' hidden states.
        :param word_segment_emb_merger: function that merges the word segments belonging to the same word. By default,
        this is the mean of the hidden states. The function takes as input
            1) a tensor with shape: (batch, max len, max word segment per word in batch, hidden size), which contains the
            hidden vectors for each word segment grouped by word they belong to.
            2) a tensor with shape: (), which contains the mask to be applied after merging.
        """
        super().__init__()
        self._matched_embedder = MultilayerPretrainedTransformerEmbedder(model_name, layers_to_merge, max_length,
                                                                         layer_merger)

        self.ws_embedding_merger = get_combiner(word_segment_emb_merger)

    @overrides
    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()

    @overrides
    def forward(
            self,
            token_ids: torch.LongTensor,
            mask: torch.BoolTensor,
            offsets: torch.LongTensor,
            wordpiece_mask: torch.BoolTensor,
            type_ids: Optional[torch.LongTensor] = None,
            segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: torch.LongTensor
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: torch.BoolTensor
            Shape: [batch_size, num_orig_tokens].
        offsets: torch.LongTensor
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: torch.BoolTensor
            Shape: [batch_size, num_wordpieces].
        type_ids: Optional[torch.LongTensor]
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: Optional[torch.BoolTensor]
            See `PretrainedTransformerEmbedder`.

        # Returns

        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self._matched_embedder(
            token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        orig_embeddings = self.ws_embedding_merger(span_embeddings, span_mask)
        # span_embeddings_sum = span_embeddings.sum(2)
        # span_embeddings_len = span_mask.sum(2)
        # # Shape: (batch_size, num_orig_tokens, embedding_size)
        # orig_embeddings = span_embeddings_sum / span_embeddings_len

        return orig_embeddings
