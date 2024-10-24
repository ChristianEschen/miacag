import yaml
import torch
import os

class ModelBuilder():
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def getFingerPrint(self, filename):
        with open(filename) as file:
            fingerprint = yaml.load(file, Loader=yaml.FullLoader)
        return fingerprint

    def getTuplesFromDict(self, dictionary):
        for d in dictionary:
            if isinstance(dictionary[d], str):
                dictionary[d] = self.convert_string_to_tuple(dictionary[d])
        return dictionary

    def convert_string_to_tuple(self, field):
        res = []
        temp = []
        for token in field.split(", "):
            num = int(token.replace("(", "").replace(")", ""))
            temp.append(num)
            if ")" in token:
                res.append(tuple(temp))
                temp = []
        return res[0]

    def get_representation_learning_model(self):
        path = self.config['model']['pretrain_model']
        from models.modules import SimSiam as m
        model = m(self.config)
        if path != "None":
            model.load_state_dict(torch.load(path))
        return model

    def get_mayby_DDP(self, model):
     #   if self.config['loaders']['val_method']['saliency'] == True:
            
        model.to(self.device)
        if self.config["cpu"] == "False":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #if config['cpu'] == "False":
        if self.config['use_DDP'] == 'True':
            if self.config['loaders']['val_method']['saliency'] == True:
                model = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[self.device] if self.config["cpu"] == "False" else None,
                        output_device=0 if self.config["cpu"] == "False" else None,
                        find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[self.device] if self.config["cpu"] == "False" else None,
                        output_device=int(os.environ['LOCAL_RANK']) if self.config["cpu"] == "False" else None,
                        find_unused_parameters=True)
                    #find_unused_parameters=True if self.config['loaders']['val_method']['saliency'] == "True" else False)
        return model

    def drop_not_encoder_modules(self, state):
        new_state = state.copy()

        for key in state:
            if key.startswith('encoder'):
                pass
            else:
                new_state.pop(key, None)
        return new_state

    def get_ImageToScalar_model(self):
        path_model = self.config['model']['pretrain_model']
        if self.config['task_type'] in ["classification", "regression"]:
            from miacag.models.modules import ImageToScalarModel as m
        elif self.config['task_type'] in ["mil_classification"]:
            from miacag.models.milmodel3d import MILModel as m
        model = m(self.config, self.device)
        model = self.get_mayby_DDP(model)

        if self.config['loaders']['mode'] in ['testing', 'prediction'] or self.config['loaders']['val_method']['saliency'] == True: # or (self.config['model']['pretrain_model'] != 'None'):
            if path_model != 'None':
                if self.config["use_DDP"] == "False":
                    model.load_state_dict(
                        torch.load(os.path.join(path_model, 'model.pt')))
                else:
                    print('loading inference model fron', path_model)
                    if self.config['cpu'] == 'True':
                        model.load_state_dict(
                            torch.load(os.path.join(path_model, 'model.pt')
                                    ,map_location='cpu'))
                    else:
                        print('loading pretrained model from:', os.path.join(path_model, 'model.pt'))
                        model.module.load_state_dict(
                            torch.load(os.path.join(path_model, 'model.pt')
                                    ,map_location='cuda:{}'.format(os.environ['LOCAL_RANK'])))
            else:
                raise ValueError('No model to load in test mode??')
        
        return model

    def get_segmentation_model(self):
        path_model = self.config['model']['pretrain_model']
        path_encoder = self.config['model']['pretrain_encoder']
        if self.config['model']['model_name'] in ['UNet2D_pretrained_enc',
                                                  'UNet3D_pretrained_enc']:
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                            dimensions=self.config['model']['dimension'],
                            encoder_name=self.config['model']['backbone'],
                            encoder_weights=None,
                            decoder_channels=self.convert_string_to_tuple(self.config['model']['decoder_channels']),
                            encoder_depth=self.config['model']['encoder_depth'],
                            in_channels=self.config['model']['in_channels'],
                            classes=self.config['model']['num_classes'])
            if path_encoder != "None":
                model.encoder.load_state_dict(torch.load(path_encoder))

        elif self.config['model']['model_name'] == 'UNet':
            import monai.networks.nets as m
            model = m.UNet(
                dimensions=2 if
                self.config['model']['dimension'] == '2D' else 3,
                in_channels=self.config['model']['in_channels'],
                out_channels=self.config['model']['num_classes'],
                channels=self.convert_string_to_tuple(
                    self.config['model']['channels']),
                strides=self.convert_string_to_tuple(self.config['model']['strides'])
                )

        elif self.config['model']['model_name'] == 'DynUNet':
            print('to be implemented')
            raise ValueError('model not implemented')

        if path_model != "None":
            model.load_state_dict(torch.load(path_model))
        return model

    def get_model(self):
        if self.config['task_type'] == "representation_learning":
            model = self.get_representation_learning_model()
        elif self.config['task_type'] in ["classification", "regression", "mil_classification"]:
            model = self.get_ImageToScalar_model()
        elif self.config['task_type'] == "segmentation":
            model = self.get_segmentation_model()
        else:
            raise ValueError('model not implemented')
        # maybe freeze backbone
        if self.config['loaders']['mode'] not in ['testing', 'prediction']:
#            print('freezing backbone', self.config['model']['freeze_backbone'])

            if self.config['model']['freeze_backbone']:
                if self.config['loaders']['val_method']['saliency'] != True:
                    if not  self.config['loaders']['only_tabular']:
                            
                        for param in model.module.encoder.parameters():
                            param.requires_grad = False
                    if len(self.config['loaders']['tabular_data_names']) > 0:
                        for param in model.module.embeddings.parameters():
                            param.requires_grad = False
                        for param in model.module.tabular_mlp.parameters():
                            param.requires_grad = False
                        for param in model.module.layer_norm_func.parameters():
                            param.requires_grad = False
                        
                    if self.config['model']['aggregation'] == 'cross_attention':
                        for param in model.module.att_pool.parameters():
                            param.requires_grad = True
                            
                #else:
                #   if self.config['model']['model_name'] in "dinov2_vits14":
                    else:
                        if self.config['task_type'] != 'regression':
                            for param in model.module.fcs.parameters():
                                param.requires_grad = True
                            for param in model.module.attention.parameters():
                                param.requires_grad = True
        return model

    def __call__(self):
        if self.config['datasetFingerprintFile'] is not None:
            self.config = self.unpack_fingerprint(self.config)
        model = self.get_model()
        return model

    def unpack_fingerprint(self, config):
        dataset_fingerprint = self.getFingerPrint(
            config['datasetFingerprintFile'])
        config['loaders']['pixdim_height'] = \
            dataset_fingerprint['original_spacing'][0]
        config['loaders']['pixdim_width'] = \
            dataset_fingerprint['original_spacing'][1]
        config['loaders']['pixdim_depth'] = \
            dataset_fingerprint['original_spacing'][2]
        if self.config['task_type'] == 'segmentation':
            if config['loaders']['Crop_height'] is None:
                config['loaders']['Crop_height'] = dataset_fingerprint['patch_size'][0]
            if config['loaders']['Crop_width'] is None:
                config['loaders']['Crop_width'] = dataset_fingerprint['patch_size'][1]
            if config['loaders']['Crop_depth'] is None:
                config['loaders']['Crop_depth'] = dataset_fingerprint['patch_size'][2]
            config['loaders']['batchSize'] = dataset_fingerprint['batch_size']

            configuration = config['model'].copy()
            configuration['spatial_dims'] = configuration['dimension']
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
            del configuration['dimension']
            del configuration['classes']
            del configuration['strides_temp']

            return configuration
        else:
            return config