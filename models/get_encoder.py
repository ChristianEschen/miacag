from monai.networks import nets
from torch import nn
import torch


def get_encoder(config):
    if config['loaders']['mode'] != 'testing':
        pretrained = config['model']['pretrained']
    else:
        pretrained = False

    # Get model
    if config['model']['backbone'] == 'r3d_18':
        model = nets.torchvision_fc.models.video.resnet.r3d_18(
            pretrained=pretrained)
        in_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-2])
        # model.fc = nn.Identity()
    elif config['model']['backbone'] == 'r2plus1d_18':
        model = nets.torchvision_fc.models.video.resnet.r2plus1d_18(
            pretrained=pretrained)
        in_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-2])
    elif config['model']['backbone'] == 'x3d_l':
        print('not implemented jet')
    elif config['model']['backbone'] == 'x3d_s':
        import pytorchvideo.models.x3d as x3d
        model = x3d.create_x3d()  # default args creates x3d_s
        in_features = model.blocks[-1].proj.in_features
        model = nn.Sequential(
            *(list(model.blocks[:-1].children()) +
              list(model.blocks[-1].children())[:-3]))
    # Not image models
    elif config['model']['backbone'] == 'linear':
        from models.backbone_encoders.tabular.base_encoders \
            import LinearEncoder
        model = LinearEncoder(config['model']['incomming_features'])
        in_features = config['model']['incomming_features']

    elif config['model']['backbone'] == 'mlp':
        from models.mlps import projection_MLP
        in_features = 128
        model = projection_MLP(config['model']['incomming_features'],
                               in_features)
    else:
        raise ValueError('not implemented')
    return model, in_features


def modelsRequiredPermute():
    model_list = ['r3d_18', 'r2plus1d_18', 'x3d_l', 'x3d_s']
    return model_list
