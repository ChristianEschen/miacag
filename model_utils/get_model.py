import importlib
import yaml
import torch
import os


def getFingerPrint(filename):
    with open(filename) as file:
        fingerprint = yaml.load(file, Loader=yaml.FullLoader)
    return fingerprint


def getTuplesFromDict(dictionary):
    for d in dictionary:
        if isinstance(dictionary[d], str):
            dictionary[d] = convert_string_to_tuple(dictionary[d])
    return dictionary


def convert_string_to_tuple(field):
    res = []
    temp = []
    for token in field.split(", "):
        num = int(token.replace("(", "").replace(")", ""))
        temp.append(num)
        if ")" in token:
            res.append(tuple(temp))
            temp = []
    return res[0]


def get_model(config):
    if config['model_name'] == 'UNet3D':
        m = importlib.import_module('models.unet3d.model')
        class_name_tag = 'UNet3D'
        model_clazz = getattr(m, class_name_tag)
        configuration = config['model']
    elif config['model_name'] == 'DYNUNet3D':
        import monai.networks.nets as m
        class_name_tag = 'DynUNet'
        model_clazz = getattr(m, class_name_tag)
        dataset_fingerprint = getFingerPrint(config['dataset_fingerprint'])
        config['loaders']['pixdim_height'] = \
            dataset_fingerprint['original_spacing'][0]
        config['loaders']['pixdim_width'] = \
            dataset_fingerprint['original_spacing'][1]
        config['loaders']['pixdim_depth'] = \
            dataset_fingerprint['original_spacing'][2]

        if config['loaders']['height'] is None:
            config['loaders']['height'] = dataset_fingerprint['patch_size'][0]
        if config['loaders']['width'] is None:
            config['loaders']['width'] = dataset_fingerprint['patch_size'][1]
        if config['loaders']['depth'] is None:
            config['loaders']['depth'] = dataset_fingerprint['patch_size'][2]
        config['loaders']['batchSize'] = dataset_fingerprint['batch_size']

        configuration = config['model'].copy()
        configuration['spatial_dims'] = configuration['dimensions']
        configuration['out_channels'] = configuration['classes']
        configuration['kernel_size'] = dataset_fingerprint['conv_kernel_sizes']
        configuration['strides_temp'] = dataset_fingerprint['pool_op_kernel_sizes']
        configuration['strides'] = []
        for i in range(0, len(configuration['strides_temp'])):
            if i == 0:
                configuration['strides'].append([1, 1, 1])
                configuration['strides'].append(configuration['strides_temp'][i])
            else:
                configuration['strides'].append(configuration['strides_temp'][i])
        configuration['upsample_kernel_size'] = configuration['strides'][1:]
        configuration['norm_name'] = "instance"
        configuration['deep_supr_num'] = 2
        configuration['res_block'] = False
        del configuration['dimensions']
        del configuration['classes']
        del configuration['strides_temp']
    elif config['model_name'] == 'UNet2D':
        import monai.networks.nets as m
        class_name_tag = 'UNet'
        model_clazz = getattr(m, class_name_tag)
        configuration = config['model'].copy()
        configuration = getTuplesFromDict(configuration)
        configuration['out_channels'] = configuration['classes']
        del configuration['classes']
    elif config['model_name'][3:] == 'csn_152_':
        m = importlib.import_module('models.csn.model')
        class_name_tag = config['model_name'][3:]
        model_clazz = getattr(m, class_name_tag)
        configuration = config['model'].copy()
        del configuration['in_channels']
        configuration['name'] = config['model_name']
    elif config['model_name'] == 'CO_CLR':
        if 'model_path' in config:
            m = importlib.import_module('models.CoCLR.classification')
            model_clazz = getattr(m, 'LinearClassifier')
            configuration = config['model']
            model = model_clazz(**configuration)
            if config['model_path'] != "":
                model.backbone.load_state_dict(
                    torch.load(os.path.join(config['model_path'], 'model.pt')))
            return model
        else:
            m = importlib.import_module('models.CoCLR.pretrain')
            model_clazz = getattr(m, 'InfoNCE')
            configuration = {}
            configuration['input_channel'] = config['model']['in_channels']
    elif config['model_name'] == 'SimSiam':
        m = importlib.import_module('models.SimSiam.simsiam2d')
        model_clazz = getattr(m, 'SimSiam')
        configuration = config['model']
        model = model_clazz(**configuration)
    else:
        raise ValueError(
            "model is not implemented%s" % repr(config['model_name']))

    model = model_clazz(**configuration)
    return model
