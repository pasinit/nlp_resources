from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer

from nlp_tools.allen_data.iterators import get_bucket_iterator
from nlp_tools.data_io.datasets import TokenizedSentencesDataset


def test_possible_labels(languages, paths):
    pass

def test_tokenized_sentences_dataset():
    sentences = ["this is a simple sentence .".split(),
                 "Claudio Monteverdi ( 15 May 1567 – 29 November 1643 ) was an Italian composer .".split(),
                 "string player and maestro di cappella .".split(),
                 "A composer of both secular and sacred music , and a pioneer in the development of opera , he is considered a transitional figure .".split(),
                 "Between the Renaissance and the Baroque periods of music history ." .split(),
                 "He was a court musician in Mantua ( c. 1590 – 1613 ) , and then maestro di cappella at St Mark's Basilica in the Republic of Venice .".split(),
                 "His surviving music includes nine books of madrigals , in the tradition of earlier".split()]
    indexer = PretrainedTransformerMismatchedIndexer("bert-large-cased")
    dataset = TokenizedSentencesDataset(sentences, indexer)
    dataset.index_with(Vocabulary())
    iterator = get_bucket_iterator(dataset, 20, is_trainingset=False)

    for elem in iterator:
        print(elem)

if __name__ == "__main__":
    # lemma2synsets, mfs_dictionary, label_vocab = get_data(sense_inventory, langs, mfs_file)
    test_tokenized_sentences_dataset()