import yaml
import os
from datetime import datetime
import socket
import numpy as np

class NumpySafeDumper(yaml.SafeDumper):
    def represent_data(self, data):
        if isinstance(data, np.ndarray):
            return self.represent_list(data.tolist())
        elif isinstance(data, np.generic):
            return self.represent_data(data.item())
        return super().represent_data(data)
def maybe_create_tensorboard_logdir(config):
    if config['create_tensorboard_timestamp']:
        config['output_directory'] = os.path.join(
            config['output_directory'],
            datetime.now().strftime('%b%d_%H-%M-%S') +
            '_' + socket.gethostname())
    return config



def load_config(config_path, configs={}):
    # get config file
    config = _load_config_yaml(config_path)
    config.update(configs)
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def save_config(writer, config, outfile='config.yaml'):
    yaml_file = os.path.join(writer.log_dir, outfile)
    with open(yaml_file, 'w') as outfile:
        yaml.dump(config, outfile, Dumper=NumpySafeDumper, default_flow_style=False)
