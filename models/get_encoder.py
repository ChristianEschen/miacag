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
        path = '/home/gandalf/MIA/models/torchhub/X3D_L.pyth'
        model = torch.hub.load("/home/gandalf/pytorchvideo-main",
                               source="local",
                               model=config['model']['backbone'],
                               pretrained=False)

        model.load_state_dict(torch.load(path)['model_state'])
        in_features = model.blocks[-1].proj.in_features
        model = nn.Sequential(
            *(list(model.blocks[:-1].children()) +
              list(model.blocks[-1].children())[:-3]))
    elif config['model']['backbone'] == 'x3d_s':
        model = torch.hub.load("/home/gandalf/pytorchvideo-main",
                               source="local",
                               model=config['model']['backbone'],
                               pretrained=pretrained)
        in_features = model.blocks[-1].proj.in_features
        model = nn.Sequential(
            *(list(model.blocks[:-1].children()) +
              list(model.blocks[-1].children())[:-3]))
    else:
        raise ValueError('not implemented')
    return model, in_features


def modelsRequiredPermute():
    model_list = ['r3d_18', 'r2plus1d_18', 'x3d_l', 'x3d_s']
    return model_list
