import unittest

from to_be_removed.nlp_resources.twenty_newsgroup import TwentyNewsgroup


class Test20Newsgroup(unittest.TestCase):

    def test_encoding(self):
        import random
        random.seed(24)
        if __name__ == "__main__":
            ng = TwentyNewsgroup("/home/tommaso/Documents/data/resources/data/20news-bydate/20news-bydate-train-clean",
                                 "/home/tommaso/Documents/data/resources/data/20news-bydate/20news-bydate-test-clean")
            docid2text = ng.train_id2text
            docid2indices = ng.train_id2encoded_text
            docid2label = ng.train_id2label
            docid2labelidx = ng.train_id2encoded_label
            id2word = ng.id2word
            ids = list(docid2text.keys())
            for i in ids:
                text = docid2text[i]
                indices = docid2indices[i]
                rec_text = [id2word[x] for x in indices]
                self.assertEqual(text == rec_text, " ".join(text) + "\n" + " ".join(rec_text))
                label = docid2label[i]
                enc_label = docid2labelidx[i]
                self.assertEqual(ng.labels[label] == enc_label, "{}\t{}".format(ng.labels[label], enc_label))


if __name__ == "__main__":
    Test20Newsgroup().test_encoding()