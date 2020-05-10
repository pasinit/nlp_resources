import os

from nlp_tools.nlp_utils.utils import get_pos_from_key

RESOURCES_DIR="resources/"

def get_wnoffset2bnoffset():
    offset2bn = __load_reverse_multimap(os.path.join(RESOURCES_DIR, "mappings/all_bn_wn.txt"))
    new_offset2bn = {"wn:" + offset: bns for offset, bns in offset2bn.items()}
    return new_offset2bn


def get_bnoffset2wnoffset():
    return __load_multimap(os.path.join(RESOURCES_DIR, "mappings/all_bn_wn.txt"), value_transformer=lambda x: "wn:" + x)


def get_wnkeys2bnoffset():
    return __load_reverse_multimap(os.path.join(RESOURCES_DIR, "mappings/all_bn_wn_keys.txt"),
                                   key_transformer=lambda x: x.replace("%5", "%3"))


def get_bnoffset2wnkeys():
    return __load_multimap(os.path.join(RESOURCES_DIR, "mappings/all_bn_wn_key.txt"), value_transformer=lambda x: x.replace("%5", "%3"))


def get_wnoffset2wnkeys():
    offset2keys = dict()
    with open("/opt/WordNet-3.0/dict/index.sense") as lines:
        for line in lines:
            fields = line.strip().split(" ")
            keys = offset2keys.get(fields[1], set())
            keys.add(fields[0].replace("%5", "%3"))
            offset2keys[fields[1]] = keys
    return offset2keys


def get_wnkeys2wnoffset():
    key2offset = dict()
    with open("/opt/WordNet-3.0/dict/index.sense") as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0].replace("%5", "%3")
            pos = get_pos_from_key(key)
            key2offset[key] = ["wn:" + fields[1] + pos]
    return key2offset


def __load_multimap(path, key_transformer=lambda x: x, value_transformer=lambda x: x):
    bnoffset2wnkeys = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnoffset = fields[0]
            bnoffset2wnkeys[key_transformer(bnoffset)] = [value_transformer(x) for x in fields[1:]]
    return bnoffset2wnkeys


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
