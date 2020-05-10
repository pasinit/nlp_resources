from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.nn.util import move_to_device
from torch.nn import CrossEntropyLoss, Module
from torch.optim.adam import Adam
from tqdm import tqdm
import torch
from nlp_resources.allen_data.iterators import AllenWSDDatasetBucketIterator
from nlp_resources.data_io.data_utils import Lemma2Synsets
from nlp_resources.data_io.datasets import wn_sensekey_vocabulary, WSDDataset
from nlp_resources.nlp_models.huggingface_wrappers import GenericHuggingfaceWrapper
from torch import nn

class SimpleWSDModel(Module):
    def __init__(self, model_name, device, **kwargs):

        super().__init__()
        self.text_embedder = PretrainedTransformerMismatchedEmbedder(model_name).cuda().eval()

        self.wsd_head = nn.Linear(1024, 117660)

    def forward(self, tokens, **kwargs):
        with torch.no_grad():
            out = self.text_embedder(**tokens)
        return self.wsd_head(out)


if __name__ == "__main__":
    lemma2synsets = Lemma2Synsets.sensekeys_from_wordnet_dict()
    label_vocab = wn_sensekey_vocabulary()

    label_mapper = None
    indexer = PretrainedTransformerMismatchedIndexer("bert-large-cased")

    dataset = WSDDataset(
        "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
        lemma2synsets,
        label_mapper, indexer, label_vocab)
    dataset.index_with(Vocabulary())
    iterator = AllenWSDDatasetBucketIterator.get_bucket_iterator(dataset, 1000)
    model = SimpleWSDModel("bert-large-cased", "cuda")
    model.cuda()
    optimizer = Adam(model.parameters())
    loss = CrossEntropyLoss().cuda()
    count = 0
    for epoch in range(10):
        bar = tqdm(iterator)
        for batch in bar:
            count += batch["tokens"]["tokens"]["token_ids"].shape[0]
            optimizer.zero_grad()
            logits = model(move_to_device(batch["tokens"]["tokens"], 0))
            logits = logits[batch["label_ids"] != 0]
            label_ids = batch["label_ids"][batch["label_ids"] != 0]
            l = loss(logits, label_ids.cuda())
            l.backward()
            optimizer.step()
            bar.set_postfix({"loss":l.item()})
        print("PORCO DIO")