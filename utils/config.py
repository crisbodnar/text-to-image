import yaml
from easydict import EasyDict


def config_from_yaml(file_path):
    with open(file_path, "r") as f:
        return EasyDict(yaml.load(f))
