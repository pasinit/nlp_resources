import logging
import re
from collections import OrderedDict, Counter
from typing import List, Dict, Union

import numpy as np
from allennlp.data import TokenIndexer, Instance, Token
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.fields import TextField, MetadataField, ArrayField
from lxml import etree
from torchtext.vocab import Vocab
from tqdm import tqdm

from nlp_resources.data_io.mapping_utils import get_wnoffset2bnoffset
from nlp_resources.nlp_utils.utils import get_pos_from_key, get_simplified_pos

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
WORDNET_DICT_PATH = "/opt/WordNet-3.0/dict/index.sense"


def wnoffset_vocabulary():
    offsets = list()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            offsets.append(offset)
    return LabelVocabulary(Counter(sorted(offsets)), specials=["<pad>", "<unk>"])


def bnoffset_vocabulary():
    # with open("resources/vocabularies/bn_vocabulary.txt") as lines:
    #     bnoffsets = [line.strip() for line in lines]
    wn2bn = get_wnoffset2bnoffset()
    offsets = set()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            bnoffset = wn2bn[offset]
            offsets.update(bnoffset)
    return LabelVocabulary(Counter(sorted(offsets)), specials=["<pad>", "<unk>"])


def wn_sensekey_vocabulary():
    with open(WORDNET_DICT_PATH) as lines:
        keys = [line.strip().split(" ")[0].replace("%5", "%3") for line in lines]
    return LabelVocabulary(Counter(sorted(keys)), specials=["<pad>", "<unk>"])


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


class WSDDataset(AllennlpDataset):
    def __init__(self, paths: Union[list, str], lemma2synsets: Dict[str, List[str]],
                 label_mapper: Union[Dict[str, str], None], indexer: TokenIndexer,
                 label_vocab: Vocab,
                 **kwargs):
        self.key2goldid = label_mapper
        self.lemma2synsets = lemma2synsets
        self.indexers = {"tokens": indexer}
        self.label_vocab = label_vocab
        self.pad_token_id = indexer._tokenizer.pad_token_id
        examples = self.load_examples(paths, indexer)

        super().__init__(examples, **kwargs)

    def load_gold_file(self, gold_file):
        key2gold = dict()
        with open(gold_file) as lines:
            for line in lines:
                fields = re.split("\s", line.strip())
                key, *gold = fields
                if self.key2goldid is not None and len(self.key2goldid) > 0:
                    gold = [self.key2goldid.get(g, self.key2goldid.get(g.replace("%5", "%3"), [None])) for g in gold]
                    gold = [x for y in gold for x in y]
                else:
                    gold = [x.replace("%5", "%3") for x in gold]
                key2gold[key] = gold
        return key2gold

    def load_examples(self, paths, indexer):
        self.start = 0
        if type(paths) != list:
            paths = [paths]
        all_examples = list()
        for i, file_path in enumerate(paths):
            gold_file = file_path.replace(".data.xml", ".gold.key.txt")
            tokid2gold = self.load_gold_file(gold_file)
            aux = self.load_xml(tokid2gold, file_path, indexer)
            all_examples.extend(aux)
        return all_examples

    def load_xml(self, tokid2gold, file_path, tokenizer):
        examples = list()
        for _, sentence in tqdm(etree.iterparse(file_path, tag="sentence"), desc="reading {}".format(file_path)):
            words = list()
            lemmaposs = list()
            ids = list()
            labels = list()
            for elem in sentence:
                if elem.text is None:
                    continue
                words.append(elem.text)

                if elem.tag == "wf" or elem.attrib["id"] not in tokid2gold:
                    ids.append(None)
                    labels.append("")
                    lemmaposs.append("")
                else:
                    ids.append(elem.attrib["id"])
                    labels.append(tokid2gold[elem.attrib["id"]])
                    lemmaposs.append(elem.attrib["lemma"].lower() + "#" + get_simplified_pos(elem.attrib["pos"]))

            if any(x is not None for x in ids):
                unique_token_ids = list(range(self.start, self.start + len([x for x in ids if x is not None])))
                examples.append(
                    self.text_to_instance(unique_token_ids, words, lemmaposs, ids, np.array(labels)))
                self.start += len(unique_token_ids)  # unique_token_ids[-1] + 1
            # if len(examples) == 100:
            #     break
        return examples

    def text_to_instance(self, unique_token_ids: List[int],
                         input_words: List[str],
                         input_lemmapos: List[str],
                         input_ids: List[str],
                         labels: np.ndarray) -> Instance:
        input_words_field = TextField([Token(x) for x in input_words], self.indexers)
        fields = {"tokens": input_words_field}
        instance_ids = [x for x in input_ids if x is not None][0]

        instance_ids = MetadataField(instance_ids)
        cache_instance_id = MetadataField(unique_token_ids)
        fields["instance_ids"] = instance_ids
        fields["cache_instance_ids"] = cache_instance_id
        id_field = MetadataField(input_ids)
        fields["ids"] = id_field

        lemmapos_field = MetadataField(input_lemmapos)
        fields["lemmapos"] = lemmapos_field

        if labels is None:
            labels = np.zeros(len(input_words))

        label_ids = []
        for labels_for_instance in labels:
            if len(labels_for_instance) < 1:
                label_ids.append(self.label_vocab.get_idx("<pad>"))
            else:
                l = labels_for_instance[0]
                label_ids.append(
                    self.label_vocab.get_idx(l) if l in self.label_vocab.stoi
                    else self.label_vocab.get_idx(
                        "<unk>"))
        label_field = ArrayField(
            array=np.array(label_ids).astype(np.int32),
            dtype=np.long)
        assert np.sum(np.array(label_ids) != 0) == len(cache_instance_id.metadata)
        fields["label_ids"] = label_field
        fields["labels"] = MetadataField([ls for ls in labels if ls is not None])

        labeled_token_indices = np.array([i for i, l in enumerate(labels) if len(l) > 0],
                                         dtype=np.int64)  # np.argwhere(labels != "").flatten().astype(np.int64)
        fields["labeled_token_indices"] = MetadataField(labeled_token_indices)

        labeled_lemmapos = MetadataField(np.array(input_lemmapos)[labeled_token_indices])
        fields["labeled_lemmapos"] = labeled_lemmapos
        possible_labels = list()
        for i in range(len(input_lemmapos)):
            if input_ids[i] is None:
                continue
            classes = self.lemma2synsets.get(input_lemmapos[i], [None])
            classes = np.array(list(classes))

            possible_labels.append(classes)

        assert len(labeled_lemmapos) == len(labeled_token_indices) == len(possible_labels)
        possible_labels_field = MetadataField(possible_labels)
        fields["possible_labels"] = possible_labels_field

        return Instance(fields)
