from monai.networks import nets
from torch import nn


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
    else:
        raise ValueError('not implemented')
    return model, in_features


def modelsRequiredPermute():
    model_list = ['r3d_18']
    return model_list
