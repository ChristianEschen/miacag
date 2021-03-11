import yaml


def load_config(config_arg):
    config = _load_config_yaml(config_arg.config)
    config.update(vars(config_arg))
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
