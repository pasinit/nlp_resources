from nlp_tools.data_io.data_utils import MultilingualLemma2Synsets
from nlp_tools.data_io.datasets import WSDDataset
from nlp_tools.nlp_models.multilayer_pretrained_transformer_mismatched_embedder import \
    MultilayerPretrainedTransformerMismatchedEmbedder
from nlp_tools.nlp_utils.utils import get_simplified_pos


def inventory_from_bn_mapping(langs=("en"), **kwargs):
    lang2inventory = dict()
    for lang in langs:
        lemmapos2gold = dict()
        with open("resources/evaluation_framework_3.0/inventories/inventory.{}.withgold.txt".format(lang)) as lines:
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

if __name__ == "__main__":
    model_name = "xlm-roberta-large"
    embedder = MultilayerPretrainedTransformerMismatchedEmbedder(model_name, layers_to_merge=[-1,-2,-3,-4])
    inventory = inventory_from_bn_mapping(["en", "it", "es", "fr", "de"])
    WSDDataset(paths, lemma2synsets=inventory, label_mapper=None,
               indexer=indexer, label_vocab=label_vocab)