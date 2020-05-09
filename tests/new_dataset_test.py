from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer

from nlp_resources import data_io as mapping_utils
from nlp_resources.allen_data.iterators import AllenWSDDatasetBucketIterator
from nlp_resources.data_io.data_utils import Lemma2Synsets
from nlp_resources.data_io.datasets import WSDDataset, wn_sensekey_vocabulary

mapping_utils.RESOURCES_DIR = "/home/tommaso/dev/PycharmProjects/WSDframework/resources"

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
    # model = AutoModel.from_pretrained("bert-large-cased")
    # model.cpu()
    # linear = Linear(1024, 117660)
    # linear.cpu()
    for x in iterator:
        aux = x["tokens"]["tokens"]["token_ids"]
        print(aux.shape[0]*aux.shape[1])
