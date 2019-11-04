from collections import OrderedDict

from allennlp.data import DatasetReader, TokenIndexer, Instance
from typing import Callable, List, Dict, Iterable

from torchtext.vocab import Vocab


class LabelVocabulary(Vocab):
    def __init__(self, counter, **kwargs):
        super().__init__(counter, **kwargs)
        self.itos = OrderedDict()
        for s, i in self.stoi.items():
            self.itos[i] = s

    def get_string(self, idx):
        return self.itos.get(idx, None)

    def get_idx(self, token):
        return self[token]

    @classmethod
    def vocabulary_from_gold_key_file(cls, gold_key, key2wnid_path=None, key2bnid_path=None):
        key2id = None
        if key2bnid_path:
            key2id = load_bn_key2id_map(key2bnid_path)
        elif key2wnid_path:
            key2id = load_wn_key2id_map(key2wnid_path)
        labels = Counter()
        with open(gold_key) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                golds = fields[1:]
                if key2id is not None:
                    golds = [key2id[g] for g in golds]
                labels.update(golds)
        return LabelVocabulary(labels)

class AllenWSDDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_vocab:LabelVocabulary=None):
        super().__init__(lazy=False)
        assert token_indexers is not None and label_vocab is not None
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.label_vocab = label_vocab

    def _read(self, file_path: str) -> Iterable[Instance]:
        gold_file = file_path.replace(".data.xml", ".gold.key.txt")
        tokid2gold = self.load_gold_file(gold_file)
        yield from self.load_xml(tokid2gold, file_path)

    def load_gold_file(self, gold_file):
        key2gold = dict()
        with open(gold_file) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key, *gold = fields
                key2gold[key] = gold
        return key2gold

    def load_xml(self, tokid2gold, file_path):
        root = etree.parse(file_path)
        for sentence in root.findall("./text/sentence"):
            words = list()
            lemmaposs = list()
            ids = list()
            labels = list()
            for elem in sentence:
                words.append(Token(elem.text, lemma_=elem.attrib["lemma"], pos_=elem.attrib["pos"]))
                lemmaposs.append(elem.attrib["lemma"] + "#" + get_simplified_pos(elem.attrib["pos"]))
                if elem.tag == "wf":
                    ids.append(None)
                    labels.append("")
                else:
                    ids.append(elem.attrib["id"])
                    labels.append(tokid2gold[elem.attrib["id"]])

            yield self.text_to_instance(words, lemmaposs, ids, np.array(labels))

    def text_to_instance(self, input_words: List[Token], input_lemmapos: List[str], input_ids: List[str],
                         labels: np.ndarray) -> Instance:


        input_words_field = TextField(input_words, self.token_indexers)
        fields = {"tokens": input_words_field}

        id_field = MetadataField(input_ids)
        fields["ids"] = id_field

        words_field = MetadataField([t.text for t in input_words_field])
        fields["words"] = words_field

        lemmapos_field = MetadataField(input_lemmapos)
        fields["lemmapos"] = lemmapos_field

        if labels is None:
            labels = np.zeros(len(input_words_field))
        label_field = ArrayField(array=np.array([self.label_vocab.get_idx(l[0] if type(l) == list else l )  for l in  labels], dtype=np.long))
        fields["labels"] = label_field

        return Instance(fields)