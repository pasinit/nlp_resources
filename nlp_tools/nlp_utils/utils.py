def get_pos_from_key(key):
    """
    assumes key is in the wordnet key format, i.e., 's_gravenhage%1:15:00:
    :param key: wordnet key
    :return: pos tag corresponding to the key
    """
    numpos = key.split("%")[-1][0]
    if numpos == "1":
        return "n"
    elif numpos == "2":
        return "v"
    elif numpos == "3" or numpos == "5":
        return "a"
    else:
        return "r"


def get_universal_pos(simplified_pos):
    if simplified_pos == "n":
        return "NOUN"
    if simplified_pos == "v":
        return "VERB"
    if simplified_pos == "a":
        return "ADJ"
    if simplified_pos == "r":
        return "ADV"
    return ""

def get_simplified_pos(long_pos):
    long_pos = long_pos.lower()
    if long_pos.startswith("n") or long_pos.startswith("propn"):
        return "n"
    elif long_pos.startswith("adj") or long_pos.startswith("j"):
        return "a"
    elif long_pos.startswith("adv") or long_pos.startswith("r"):
        return "r"
    elif long_pos.startswith("v"):
        return "v"
    return "o"