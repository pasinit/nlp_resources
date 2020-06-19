from allennlp.data import Vocabulary, AllennlpDataset, DataLoader
from allennlp.data.samplers import BasicBatchSampler, BucketBatchSampler, MaxTokensBatchSampler
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from torch.utils.data.sampler import SequentialSampler

from nlp_tools.allen_data.iterators import get_bucket_iterator
from nlp_tools.data_io.data_utils import MultilingualLemma2Synsets
from nlp_tools.data_io.datasets import WSDDataset
from nlp_tools.nlp_models.multilayer_pretrained_transformer_mismatched_embedder import \
    MultilayerPretrainedTransformerMismatchedEmbedder
from nlp_tools.nlp_utils.utils import get_simplified_pos


def inventory_from_bn_mapping(langs=("en",), **kwargs):
    lang2inventory = dict()
    for lang in langs:
        lemmapos2gold = dict()
        with open("/home/tommaso/dev/PycharmProjects/WSDframework/resources/evaluation_framework_3.0/inventories/inventory.{}.withgold.txt".format(lang)) as lines:
            for line in lines:
                fields = line.strip().lower().split("\t")
                if len(fields) < 2:
                    continue
                lemma, pos = fields[0].split("#")
                pos = get_simplified_pos(pos)
                lemmapos = lemma + "#" + pos  # + "#" + lang
                synsets = fields[1:]
                old_synsets = lemmapos2gold.get(lemmapos, set())
                old_synsets.update(synsets)
                lemmapos2gold[lemmapos] = old_synsets
        lang2inventory[lang] = lemmapos2gold
    return MultilingualLemma2Synsets(**lang2inventory)

def __load_reverse_multimap(path, key_transformer=lambda x: x, value_transformer=lambda x: x):
    sensekey2bnoffset = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnid = fields[0]
            for key in fields[1:]:
                offsets = sensekey2bnoffset.get(key, set())
                offsets.add(value_transformer(bnid))
                sensekey2bnoffset[key_transformer(key)] = offsets
    for k, v in sensekey2bnoffset.items():
        sensekey2bnoffset[k] = list(v)
    return sensekey2bnoffset

if __name__ == "__main__":
    encoder_name = "xlm-roberta-large"
    print("loading indexer")
    indexer = PretrainedTransformerMismatchedIndexer(encoder_name)
    print("loading embedder")
    embedder = MultilayerPretrainedTransformerMismatchedEmbedder(encoder_name, layers_to_merge=[-1,-2,-3,-4])
    print("loading inventory")
    inventory = inventory_from_bn_mapping(("en", "it", "es", "fr", "de"))
    paths = {"en":["/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/en_training_data/semcor/semcor.data.xml"]}
                   # "/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/en_training_data/wngt_michele/wngt_michele_examples/wngt_michele_examples.data.xml",
                   # "/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/en_training_data/wngt_michele/wngt_michele_glosses/wngt_michele_glosses.data.xml"]}
    print("loading datasets")
    label_mapper = __load_reverse_multimap("/home/tommaso/dev/PycharmProjects/WSDframework/resources/mappings/all_bn_wn_keys.txt")
    dataset = WSDDataset(paths, lemma2synsets=inventory, label_mapper=label_mapper,
               indexer=indexer, label_vocab=None)
    dataset.index_with(Vocabulary())
    print("batching...")
    # iterator = get_bucket_iterator(dataset, 1000, is_trainingset=False)
    # sampler = SequentialSampler(data_source=dataset)
    # batch_sampler = BasicBatchSampler(sampler, batch_size=2, drop_last=False)
    batch_sampler = MaxTokensBatchSampler(dataset, 1000, ["tokens"])
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
    for batch in data_loader:
        print(batch["tokens"]["tokens"]["token_ids"].shape)
        # outputs = embedder(**batch["tokens"]["tokens"])
        # for i, (ids, labels) in enumerate(zip(batch["ids"], batch["labels"])):
        #     vectors = outputs[i]
        #     ids = [x for x in ids if x != None]
        #     labels = [x for x in labels if x != ""]
        #     for instance_id, instance_label, instance_vector in zip(ids, labels, vectors[:len(ids)]):
        #         print(instance_id, instance_label, instance_vector.detach().cpu())
        #
        # print(outputs)
        # break
    print("ok")


