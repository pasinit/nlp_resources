from collections import OrderedDict, Counter

from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Instance
from typing import Callable, List, Dict, Iterable, Union, Any

from allennlp.data.dataset_readers.dataset_reader import _LazyInstances
from allennlp.data.fields import TextField, MetadataField, ArrayField
from torchtext.vocab import Vocab

from config.nlp_resources_config import load_data_config
from data_io.data_utils import Lemma2Synsets, load_bn_key2id_map, load_wn_key2id_map
import logging
import os
import re
from allennlp.data.tokenizers import Token
import numpy as np
from lxml import etree

from data_io.mapping_utils import get_wnoffset2bnoffset, get_wnoffset2wnkeys, get_wnkeys2bnoffset, \
    get_bnoffset2wnoffset, get_wnkeys2wnoffset, get_bnoffset2wnkeys
from nlp_utils.utils import get_simplified_pos, get_pos_from_key

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
                golds = [x.replace("%5", "%3") for x in fields[1:]]
                if key2id is not None:
                    golds = [key2id[g] for g in golds]
                labels.update(golds)
        return LabelVocabulary(labels, specials=["<pad>", "<unk>"])

    @classmethod
    def wnoffset_vocabulary(cls):
        offsets = list()
        data_config = load_data_config()
        with open(data_config["wordnet_path"]) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key = fields[0]
                pos = get_pos_from_key(key)
                offset = "wn:" + fields[1] + pos
                offsets.append(offset)
        return LabelVocabulary(Counter(sorted(offsets)), specials=["<pad>", "<unk>"])

    @classmethod
    def bnoffset_vocabulary(cls):
        # with open("resources/vocabularies/bn_vocabulary.txt") as lines:
        #     bnoffsets = [line.strip() for line in lines]
        wn2bn = get_wnoffset2bnoffset()
        offsets = set()
        data_config = load_data_config()
        with open(data_config["wordnet_path"]) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key = fields[0]
                pos = get_pos_from_key(key)
                offset = "wn:" + fields[1] + pos
                bnoffset = wn2bn[offset]
                offsets.update(bnoffset)
        return LabelVocabulary(Counter(sorted(offsets)), specials=["<pad>", "<unk>"])

    @classmethod
    def wn_sensekey_vocabulary(cls):
        data_config = load_data_config()
        with open(data_config["wordnet_path"]) as lines:
            keys = [line.strip().split(" ")[0].replace("%5", "%3") for line in lines]
        return LabelVocabulary(Counter(sorted(keys)), specials=["<pad>", "<unk>"])


@DatasetReader.register("wsd_xml")
class AllenWSDDatasetReader(DatasetReader):
    def __init__(self, sense_inventory, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_vocab: LabelVocabulary = None, lemma2synsets=None,
                 # key2goldid: Dict[str, str] = None,
                 max_sentence_len: int = 64,
                 sliding_window_size: int = 32, gold_key_id_separator=" ",
                 lazy=False,
                 **kwargs):
        super().__init__(lazy=lazy)
        assert token_indexers is not None and label_vocab is not None and lemma2synsets is not None
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.label_vocab = label_vocab
        self.lemma2synsets = lemma2synsets
        self.key2goldid = None
        self.sense_inventory = sense_inventory
        self.max_sentence_len = max_sentence_len
        self.sliding_window_size = sliding_window_size
        self.start = 0

    def read(self, file_path: Union[str, List]) -> Iterable[Instance]:
        """
        Returns an ``Iterable`` containing all the instances
        in the specified dataset.

        If ``self.lazy`` is False, this calls ``self._read()``,
        ensures that the result is a list, then returns the resulting list.

        If ``self.lazy`` is True, this returns an object whose
        ``__iter__`` method calls ``self._read()`` each iteration.
        In this case your implementation of ``_read()`` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a ``ConfigurationError``.

        In either case, the returned ``Iterable`` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        lazy = getattr(self, 'lazy', None)

        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")

        if self._cache_directory and type(file_path) == str:
            cache_file = self._get_cache_location_for_file_path(file_path)
        else:
            cache_file = None

        if lazy:
            return _LazyInstances(lambda: self._read(file_path),
                                  cache_file,
                                  self.deserialize_instance,
                                  self.serialize_instance)
        else:
            # First we read the instances, either from a cache or from the original file.
            if cache_file and os.path.exists(cache_file):
                instances = self._instances_from_cache_file(cache_file)
            else:
                instances = self._read(file_path)

            # Then some validation.
            if not isinstance(instances, list):
                instances = [instance for instance in instances]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(
                    file_path if type(file_path) == str else ",".join(file_path)))

            # And finally we write to the cache if we need to.
            if cache_file and not os.path.exists(cache_file):
                logger.info(f"Caching instances to {cache_file}")
                self._instances_to_cache_file(cache_file, instances)

            return instances

    def _read(self, file_paths: Union[str, List]) -> Iterable[Instance]:
        self.start = 0
        if type(file_paths) != list:
            file_paths = [file_paths]
        for i, file_path in enumerate(file_paths):
            gold_file = file_path.replace(".data.xml", ".gold.key.txt")
            tokid2gold = self.load_gold_file(gold_file)
            yield from self.load_xml(tokid2gold, file_path, file_path.split("/")[-1].replace(".data.xml", ""))

    def get_goldid_by_key(self, key):
        goldid = self.key2goldid.get(key, None)
        if goldid is None:
            goldid = self.key2goldid.get(key.replace("%5", "%3"), key)
        if key == goldid:
            #     logger.warning("cannot map key {}, leaving it unchanged".format(key))
            goldid = [goldid]
        if type(goldid) != list:
            goldid = [goldid]

        return goldid

    # def get_goldid_by_key(self, key):
    #     goldid = self.key2goldid.get(key, None)
    #     if goldid is None:
    #         goldid = self.key2goldid.get(key.replace("%3", "%5"), None)
    #     return goldid
    def load_key2goldid(self, golds):
        aux = AllenWSDDatasetReader.get_label_mapper(self.sense_inventory, set(golds))
        return aux if aux is not None else {}

    def load_gold_file(self, gold_file):
        key2gold = dict()
        with open(gold_file) as lines:
            for line in lines:
                fields = re.split("\s", line.strip())
                key, *gold = fields
                if self.key2goldid is None:
                    self.key2goldid = self.load_key2goldid(gold)
                if len(self.key2goldid) > 0:
                    gold = [self.get_goldid_by_key(g) for g in gold]
                    gold = [x for y in gold for x in y]
                else:
                    gold = [x.replace("%5", "%3") for x in gold]
                key2gold[key] = gold
        return key2gold

    def load_xml(self, tokid2gold, file_path, dataset_id):
        # root = etree.parse(file_path)
        # words = list()
        # lemmaposs = list()
        # ids = list()
        # labels = list()
        for _, sentence in etree.iterparse(file_path, tag="sentence"):
            words = list()
            lemmaposs = list()
            ids = list()
            labels = list()
            for elem in sentence:
                if elem.text is None:
                    continue
                words.append(Token(elem.text))

                if elem.tag == "wf" or elem.attrib["id"] not in tokid2gold:
                    ids.append(None)
                    labels.append("")
                    lemmaposs.append("")
                else:
                    ids.append(elem.attrib["id"])
                    labels.append(tokid2gold[elem.attrib["id"]])
                    lemmaposs.append(elem.attrib["lemma"].lower() + "#" + get_simplified_pos(elem.attrib["pos"]))

            if len(words) > self.max_sentence_len:
                for w_window, lp_window, iis_window, ls_window in self.sliding_window(words, lemmaposs, ids, labels):
                    if len(w_window) > 0 and any(x is not None for x in iis_window):
                        unique_token_ids = list(
                            range(self.start, self.start + len([x for x in iis_window if x is not None])))
                        # unique_token_ids = [i+self.start for i in range(len(iis_window)) if iis_window[i] is not None]
                        # if 2354 in unique_token_ids:
                        #     print()
                        yield self.text_to_instance(unique_token_ids, w_window, lp_window, iis_window,
                                                    np.array(ls_window))
                        self.start += len(unique_token_ids)  # unique_token_ids[-1] + 1

            else:
                if any(x is not None for x in ids):
                    unique_token_ids = list(range(self.start, self.start + len([x for x in ids if x is not None])))
                    # unique_token_ids = [i + self.start for i in range(len(ids)) if ids[i] is not None]
                    # if 2354 in unique_token_ids:
                    #     print()
                    yield self.text_to_instance(unique_token_ids, words, lemmaposs, ids, np.array(labels))
                    # self.start += unique_token_ids[-1] + 1
                    self.start += len(unique_token_ids)  # unique_token_ids[-1] + 1
            # break
            # if len(words) > 0:
        #     yield self.text_to_instance(words, lemmaposs, ids, np.array(labels))
        #     print(self.start)

    def sliding_window(self, words, lemmapos, ids, labels):
        for i in range(0, len(words), self.sliding_window_size):
            w_window = words[i:i + self.max_sentence_len]
            lp_window = lemmapos[i:i + self.max_sentence_len]
            is_window = ids[i:i + self.max_sentence_len]
            ls = labels[i:i + self.max_sentence_len]
            yield w_window, lp_window, is_window, ls
            if i + self.max_sentence_len > len(words):
                break
        return

    def text_to_instance(self, unique_token_ids: List[int], input_words: List[Token], input_lemmapos: List[str],
                         input_ids: List[str],
                         labels: np.ndarray) -> Instance:
        input_words_field = TextField(input_words, self.token_indexers)
        fields = {"tokens": input_words_field}
        instance_ids = [x for x in input_ids if x is not None][0]
        instance_ids = ".".join(instance_ids.split(".")[:2])

        instance_ids = MetadataField(instance_ids)
        cache_instance_id = MetadataField(unique_token_ids)
        fields["instance_ids"] = instance_ids
        fields["cache_instance_ids"] = cache_instance_id
        id_field = MetadataField(input_ids)
        fields["ids"] = id_field

        words_field = MetadataField([t.text for t in input_words_field])
        fields["words"] = words_field

        lemmapos_field = MetadataField(input_lemmapos)
        fields["lemmapos"] = lemmapos_field

        if labels is None:
            labels = np.zeros(len(input_words_field))

        label_ids = []
        for l in labels:
            if len(l) < 1:
                label_ids.append(self.label_vocab.get_idx("<pad>"))
            else:
                label_ids.append(
                    self.label_vocab.get_idx(l[0]) if l[0] in self.label_vocab.stoi else self.label_vocab.get_idx(
                        "<unk>"))
        label_field = ArrayField(
            array=np.array(label_ids).astype(np.int32),
            dtype=np.long)
        assert np.sum(np.array(label_ids) != 0) == len(cache_instance_id.metadata)
        fields["label_ids"] = label_field
        fields["labels"] = MetadataField([ls for ls in labels if len(ls) > 0])

        labeled_token_indices = np.array([i for i, l in enumerate(labels) if len(l) > 0],
                                         dtype=np.int64)  # np.argwhere(labels != "").flatten().astype(np.int64)
        fields["labeled_token_indices"] = MetadataField(labeled_token_indices)

        labeled_lemmapos = MetadataField(np.array(input_lemmapos)[labeled_token_indices])
        fields["labeled_lemmapos"] = labeled_lemmapos
        possible_labels = list()
        for i in range(len(input_lemmapos)):
            if input_ids[i] is None:
                continue
            classes = self.lemma2synsets.get(input_lemmapos[i], [self.label_vocab.get_idx("<unk>")])
            classes = np.array(list(classes))

            possible_labels.append(classes)

        assert len(labeled_lemmapos) == len(labeled_token_indices) == len(possible_labels)
        assert len(fields["labels"].metadata) == len(labeled_lemmapos)
        possible_labels_field = MetadataField(possible_labels)
        fields["possible_labels"] = possible_labels_field

        return Instance(fields)

    @staticmethod
    def get_label_mapper(target_inventory, labels):
        label_types = set(
            ["wnoffsets" if l.startswith("wn:") else "bnoffsets" if l.startswith("bn:") else "sensekeys" for l in labels
             if l != "<pad>" and l != "<unk>"])
        if target_inventory in label_types:
            label_types.remove(target_inventory)
        if len(label_types) > 1:
            raise RuntimeError(
                "cannot handle the mapping from 2 or more label types ({}) to the target inventory {}".format(
                    ",".join(label_types), target_inventory))
        if len(label_types) == 0:
            return None
        label_type = next(iter(label_types))
        if label_type == "wnoffsets":
            if target_inventory == "bnoffsets":
                return get_wnoffset2bnoffset()
            elif target_inventory == "sensekeys":
                return get_wnoffset2wnkeys()
            return None
        elif label_type == "sensekeys" is not None:
            if target_inventory == "bnoffsets":
                return get_wnkeys2bnoffset()
            elif target_inventory == "wnoffsets":
                return get_wnkeys2wnoffset()
            else:
                return None
        else:
            if target_inventory == "wnoffsets":
                return get_bnoffset2wnoffset()
            elif target_inventory == "sensekeys":
                return get_bnoffset2wnkeys()
            else:
                raise RuntimeError("Cannot infer label type from {}".format(label_type))

    @staticmethod
    def get_dataset_with_labels_from_data(indexers: Dict[str, Any], training_data_xmls, sliding_window=32,
                                          max_sentence_token=64, gold_id_separator=" ", sense_inventory="babelnet",
                                          mfs_file=None,
                                          **kwargs):
        labels = set()
        lemma2synsetslist = list()
        for xml_path in training_data_xmls:
            gold_path = xml_path.replace("data.xml", "gold.key.txt")
            lemma2synsetslist.append(Lemma2Synsets.from_corpus_xml(xml_path))
            vocab = LabelVocabulary.vocabulary_from_gold_key_file(gold_path)
            labels.update(vocab.stoi.keys())
        key_mapper = AllenWSDDatasetReader.get_label_mapper(sense_inventory, labels)
        labels = list([list(key_mapper.get(x, {x}))[0] if key_mapper is not None else x for x in labels])
        labels = sorted(labels)
        label_vocab = LabelVocabulary(Counter(labels), specials=["<pad>", "<unk>"])
        lemma2classes = dict()
        for l2s in lemma2synsetslist:
            for lemma, synsets in l2s.items():
                all_classes = lemma2classes.get(lemma, set())
                all_classes.update(
                    [label_vocab.get_idx(y) for x in synsets for y in (key_mapper.get(x, [x]) if key_mapper else [x])])
                lemma2classes[lemma] = all_classes
        lemma2classes = Lemma2Synsets(data=lemma2classes)
        return AllenWSDDatasetReader.get_dataset(indexers, sliding_window, max_sentence_token, gold_id_separator,
                                                 label_vocab, lemma2classes, mfs_file, sense_inventory, **kwargs)

    @staticmethod
    def get_mfs_vocab(mfs_file):
        if mfs_file is None:
            return None
        mfs = dict()
        with open(mfs_file) as lines:
            for line in lines:
                fields = line.strip().lower().split("\t")
                if len(fields) < 2:
                    continue
                mfs[fields[0].lower()] = fields[1].replace("%5", "%3")
        return mfs

    @staticmethod
    def get_dataset(indexers: Dict[str, Any], sliding_window, max_sentence_token, gold_id_separator,
                    label_vocab, lemma2synsets, mfs_file, sense_inventory, **kwargs):
        reader = AllenWSDDatasetReader(sense_inventory, None, indexers, label_vocab=label_vocab,
                                       lemma2synsets=lemma2synsets,
                                       max_sentence_len=max_sentence_token,
                                       sliding_window_size=sliding_window,
                                       # key2goldid=gold_mapper,
                                       gold_key_id_separator=gold_id_separator, **kwargs)
        if label_vocab is None:
            label_vocab = reader.label_vocab
        mfs_vocab = AllenWSDDatasetReader.get_mfs_vocab(mfs_file)
        return reader, lemma2synsets, label_vocab, mfs_vocab

    @staticmethod
    def get_wnoffsets_dataset(indexers: Dict[str, Any], sliding_window=32, max_sentence_token=64,
                              gold_id_separator=" ", langs=None, mfs_file=None,
                              **kwargs):
        # if langs is not None:
        #     logger.warning("the argument langs: {} is ignored by this method.".format(",".join(langs)))
        label_vocab = LabelVocabulary.wnoffset_vocabulary()
        lemma2synsets = Lemma2Synsets.offsets_from_wn_sense_index()
        if langs is not None:
            if "en" in langs:
                langs.remove("en")
            if len(langs) > 0:
                bn2wn = get_bnoffset2wnoffset()
                bnlemma2synsets = Lemma2Synsets.from_bn_mapping(langs, **kwargs)
                for key, bns in bnlemma2synsets.items():
                    wns = [x for y in bns for x in bn2wn[y]]
                    if key in lemma2synsets:
                        lemma2synsets[key].update(wns)
                    else:
                        lemma2synsets[key] = wns

        for key, synsets in lemma2synsets.items():
            lemma2synsets[key] = [label_vocab.get_idx(l) for l in synsets]
        # key_mapper = AllenWSDDatasetReader.get_label_mapper("wnoffset", label_vocab.stoi.keys())
        return AllenWSDDatasetReader.get_dataset(indexers, sliding_window, max_sentence_token, gold_id_separator,
                                                 label_vocab, lemma2synsets, mfs_file, **kwargs)

    @staticmethod
    def get_bnoffsets_dataset(indexers: Dict[str, Any], sliding_window=32, max_sentence_token=64,
                              gold_id_separator=" ", langs=("en"), mfs_file=None, **kwargs):
        lemma2synsets = Lemma2Synsets.from_bn_mapping(langs, **kwargs)
        label_vocab = LabelVocabulary.bnoffset_vocabulary()
        for key, synsets in lemma2synsets.items():
            lemma2synsets[key] = [label_vocab.get_idx(l) for l in synsets]
        # key_mapper = AllenWSDDatasetReader.get_label_mapper("babelnet", label_vocab.stoi.keys())
        return AllenWSDDatasetReader.get_dataset(indexers, sliding_window, max_sentence_token, gold_id_separator,
                                                 label_vocab, lemma2synsets, mfs_file, **kwargs)

    @staticmethod
    def get_sensekey_dataset(indexers: Dict[str, Any], sliding_window=32, max_sentence_token=64, gold_id_separator=" ",
                             langs=None, mfs_file=None, **kwargs):
        if langs is not None:
            logger.warning(
                "[get_sensekey_dataset]: the argument langs: {} is ignored by this method.".format(",".join(langs)))
        lemma2synsets = Lemma2Synsets.sensekey_from_wn_sense_index()
        label_vocab = LabelVocabulary.wn_sensekey_vocabulary()
        for key, synsets in lemma2synsets.items():
            lemma2synsets[key] = [label_vocab.get_idx(l) for l in synsets]
        return AllenWSDDatasetReader.get_dataset(indexers, sliding_window, max_sentence_token, gold_id_separator,
                                                 label_vocab, lemma2synsets, mfs_file, **kwargs)
