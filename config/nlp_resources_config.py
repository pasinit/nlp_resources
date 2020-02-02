from allennlp.common.checks import ConfigurationError
from allennlp.common.configuration import Config, ConfigItem
import json

DATA_CONFIG = None


def load_data_config(path="resources/config.yaml"):
    global DATA_CONFIG
    if DATA_CONFIG is None:
        json_dict = json.load(path)
        if "data" not in json_dict:
            raise ConfigurationError("Cannot find data sub config in config file {}".format(path))
        data_dict = json_dict["data"]
        items = list()
        for key, value in data_dict.items():
            item = ConfigItem(name=key, annotation=str, default_value=value)
            items.append(item)
        DATA_CONFIG = Config(items)
    return DATA_CONFIG
