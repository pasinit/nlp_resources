import torch
from torch.nn import Module


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

class RobertaWrapper(Module):
    def __init__(self, model_size: str, device:str):
        super().__init__()
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.{}'.format(model_size))#, force_reload=True)
        self.roberta.eval()
        self.roberta.to(device)
        self.device = device

    def get_bpe_ids(self, sentences):
        if type(sentences) == str:
            sentences = [sentences]
        return collate_tokens([self.roberta.encode(sentence) for sentence in sentences], pad_idx=1)

    def get_words(self, bpe_ids):
        if type(bpe_ids) == str:
            bpe_ids = [bpe_ids]
        return collate_tokens([self.roberta.decode(bpe) for bpe in bpe_ids], pad_idx=1)


    def forward(self, words, align_to_words=False, return_all_hiddens=False):
        """

        :param words: if align to words we assume words to me a list of already tokenised sentences (i.e., List[List[str]])
        :param align_to_words:
        :param return_all_hiddens:
        :return:
        """
        with torch.no_grad():
            if not align_to_words:
                tokens = self.get_bpe_ids(words)
                return self.roberta.extract_features(tokens, return_all_hiddens=return_all_hiddens)
            else:
                ## TODO fare manualmente l'allinemanto segmenti - parole. Internamente usa spacy per tokenizzare ma
                ## TODO funziona solo per l'inglese e comunque si aspetta delle stringhe e non una lista di token.
                # return torch.stack([self.roberta.extract_features_aligned_to_words(sentence) for sentence in words], 0).to(self.device)
                raise Exception("not implemented yet!")

if __name__ == "__main__":
    roberta = RobertaWrapper("large", "cpu")
    hidden_states = roberta(["let's see if this model is multilingual", "vediamo se questo modello è multilingua"])
    print("cane")
