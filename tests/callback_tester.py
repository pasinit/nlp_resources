from allennlp.data import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.training import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from pytorch_transformers import AutoModel
from torch.nn import Linear
from torch.nn.modules.module import Module
from torch.utils.data.dataloader import DataLoader

from nlp_resources.data_io.data_utils import Lemma2Synsets
from nlp_resources.data_io.datasets import wn_sensekey_vocabulary, WSDDataset

class DummyModel(Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained("bert-large-cased").cpu()
        self.linear = Linear(1024, 117660).cpu()
    def forward(self, batch):
        out = self.model(batch)
        return self.linear(out[:,0])
if __name__ == "__main__":
    lemma2synsets = Lemma2Synsets.sensekeys_from_wordnet_dict()
    label_vocab = wn_sensekey_vocabulary()

    label_mapper = None
    indexer = PretrainedTransformerIndexer("bert-large-cased")
    tokenizer = PretrainedTransformerTokenizer("bert-large-cased")

    dataset = WSDDataset("/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
                         lemma2synsets,
                         label_mapper,
                         tokenizer, indexer, label_vocab)
    dataset.index_with(Vocabulary())

    data_loader = DataLoader(dataset, batch_sampler=BucketBatchSampler(dataset, 32, ["tokens"]), collate_fn=lambda x : x)

    model = DummyModel()
    optimizer = AdamOptimizer(model.named_parameters())
    trainer = GradientDescentTrainer(model,
                                     optimizer,
                                     data_loader)
    trainer.train()
    #TODO test trainer and callbacks