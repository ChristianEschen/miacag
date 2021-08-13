import yaml
import os


def load_config(config_path, configs={}):
    # get config file
    config = _load_config_yaml(config_path)
    config.update(configs)
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def save_config(writer, config):
    yaml_file = os.path.join(writer.log_dir, 'config.yaml')
    with open(yaml_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
