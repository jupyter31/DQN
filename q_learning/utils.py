import os
import yaml

def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


yaml.add_constructor("!join", join)

def read_config(config_name: str):
    config_folder = os.path.join(os.path.dirname(__file__), '..', 'q_learning', 'config')
    config_path = os.path.join(config_folder, config_name)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config