import yaml
import os
from datetime import datetime
import socket
import numpy as np
import torch

def cast_np_array_to_float(data):
    """
    Scans through the dictionary and casts any NumPy array with a single element to a float.
    
    Parameters:
    data (dict): Dictionary to scan and cast values.
    
    Returns:
    dict: Updated dictionary with NumPy arrays cast to float.
    """
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.size == 1:
            data[key] = float(value)
    
    return data

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



def cast_tensors_to_numpy(input_dict):
    result_dict = {}
    for key, value in input_dict.items():
        if torch.is_tensor(value):  # Check if the value is a tensor
            result_dict[key] = value.cpu().numpy()  # Convert tensor to numpy
        else:
            result_dict[key] = value  # Leave non-tensor values as is
    return result_dict

def load_config(config_path, configs={}):
    # get config file
    config = _load_config_yaml(config_path)
    config.update(configs)
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def save_config(writer, config, outfile='config.yaml'):
    yaml_file = os.path.join(writer.log_dir, outfile)
    config = cast_tensors_to_numpy(config)
    with open(yaml_file, 'w') as outfile:
        yaml.dump(config, outfile, Dumper=NumpySafeDumper, default_flow_style=False, sort_keys=False)
