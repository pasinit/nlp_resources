from typing import Dict

from lxml import etree

from nlp_tools.nlp_utils.utils import get_pos_from_key, get_simplified_pos

# WORDNET_DICT_PATH = "/opt/WordNet-3.0/dict/index.sense"

def load_bn_offset2bnid_map(path):
    offset2bnid = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnid = fields[0]
            for wnid in fields[1:]:
                offset2bnid[wnid] = bnid
    return offset2bnid


def load_wn_key2id_map(path):
    """
    assume the path points to a file in the same format of index.sense in WordNet dict/ subdirectory
    :param path: path to the file
    :return: dictionary from key to wordnet offsets
    """
    key2id = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            key2id[key] = ("wn:%08d" % int(fields[1])) + pos
    return key2id


def load_bn_key2id_map(path):
    """
    assumes the path points to a file with the following format:
    bnid\twn_key1\twn_key2\t...
    :param path:
    :return: a dictionary from wordnet key to bnid
    """
    key2bn = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bn = fields[0]
            for k in fields[1:]:
                key2bn[k] = bn
    return key2bn


class MultilingualLemma2Synsets:
    def __init__(self, **kwargs):
        self.lang2inventory = kwargs

    def get_inventory(self, lang):
        return self.lang2inventory.get(lang, None)

    def get(self, lexeme, lang = "en", default=None):
        if lang not in self.lang2inventory:
            return default
        return self.lang2inventory[lang].get(lexeme, default)

class Lemma2Synsets(dict):
    def __init__(self, path: str = None,
                 data: Dict = None,
                 separator="\t",
                 key_transform=lambda x: x,
                 value_transform=lambda x: x):
        """
        :param path: path to lemma 2 synset map.
        """
        assert not path or data
        super().__init__()
        if data is not None:
            self.update(data)
        else:
            with open(path) as lines:
                for line in lines:
                    fields = line.strip().split(separator)
                    key = key_transform(fields[0])
                    synsets = self.get(key, list())
                    synsets.extend([value_transform(v) for v in fields[1:]])
                    self[key] = synsets

    @staticmethod
    def load_keys(keys_path):
        key2gold = dict()
        with open(keys_path) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key2gold[fields[0]] = fields[1]
        return key2gold

    @staticmethod
    def from_corpus_xml(corpus_path, gold_transformer=lambda v: v):
        key_path = corpus_path.replace("data.xml", "gold.key.txt")
        key2gold = Lemma2Synsets.load_keys(key_path)
        # root = etree.parse(corpus_path).getroot()
        lemmapos2gold = dict()
        for _, instance in etree.iterparse(corpus_path, tag="instance", events=("start",)):
            tokenid = instance.attrib["id"]
            lemmapos = instance.attrib["lemma"].lower() + "#" + get_simplified_pos(instance.attrib["pos"])
            lemmapos2gold[lemmapos] = lemmapos2gold.get(lemmapos, set())
            lemmapos2gold[lemmapos].add(key2gold[tokenid].replace("%5", "%3"))
        for lemmapos, golds in lemmapos2gold.items():
            lemmapos2gold[lemmapos] = set(filter(lambda x: x is not None, [gold_transformer(g) for g in golds]))
        return Lemma2Synsets(data=lemmapos2gold)

    @staticmethod
    def sensekeys_from_wordnet_dict():
        from nltk.corpus import wordnet as wn
        lemmapos2gold = dict()

        for synset in wn.all_synsets():
            pos = synset.pos()
            for sense in synset.lemmas():
                sensekey = sense.key()
                lemma = sense.name()
                lexeme = lemma + "#" + pos
                golds = lemmapos2gold.get(lexeme, set())
                golds.add(sensekey)
                lemmapos2gold[lexeme] = golds
        return Lemma2Synsets(data=lemmapos2gold)

        # with open(WORDNET_DICT_PATH) as lines:
        #     for line in lines:
        #         fields = line.strip().split(" ")
        #         key = fields[0]
        #         pos = get_pos_from_key(key)
        #         lexeme = key.split("%")[0] + "#" + pos
        #         golds = lemmapos2gold.get(lexeme, set())
        #         golds.add(key)
        #         lemmapos2gold[lexeme] = golds
        # return Lemma2Synsets(data=lemmapos2gold)