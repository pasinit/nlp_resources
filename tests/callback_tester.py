from allennlp.data import Vocabulary, DataLoader, Batch
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from pytorch_transformers import AutoModel
from torch.nn import Linear

from data_io.data_utils import Lemma2Synsets
from data_io.datasets import wn_sensekey_vocabulary, WSDDataset

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

    data_loader = DataLoader(dataset, batch_sampler=BucketBatchSampler(dataset, 32, ["tokens"]), collate_fn=lambda x : Batch(x))
    model = AutoModel.from_pretrained("bert-large-cased")
    model.cuda()
    linear = Linear(1024, 117660)
    linear.cuda()
    #TODO test trainer and callbacks