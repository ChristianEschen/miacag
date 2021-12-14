from torch import nn
from models.mlps import prediction_MLP, projection_MLP
from models.get_encoder import get_encoder, modelsRequiredPermute
import torch.nn.functional as F


def maybePermuteInput(x, config):
    model_list = modelsRequiredPermute()
    if config['model']['backbone'] in model_list:
        x = x.permute(0, 1, 4, 2, 3)
        return x
    else:
        return x


class EncoderModel(nn.Module):
    def __init__(self, config):
        super(EncoderModel, self).__init__()

        self.encoder, self.in_features = get_encoder(config)

    def forward(self, x):
        x = maybePermuteInput(x, self.config)
        z = self.encoder(x)
        return z


class ClassificationModel(EncoderModel):
    def __init__(self, config):
        super(ClassificationModel, self).__init__(config)
        self.config = config
        self.fc = nn.Linear(self.in_features,
                            config['model']['num_classes'])
        self.dimension = config['model']['dimension']

    def forward(self, x):
        x = maybePermuteInput(x, self.config)
        p = self.encoder(x)
        if self.dimension in ['3D', '2D+T']:
            p = p.mean(dim=(-3, -2, -1))
        else:
            p = p.mean(dim=(-2, -1))
        p = self.fc(p)
        return p


class SimSiam(EncoderModel):
    def __init__(self, config):
        super(SimSiam, self).__init__(
            config)
        self.projector = projection_MLP(self.in_features,
                                        config['model']['feat_dim'],
                                        config['model']['num_proj_layers'],
                                        config['model']['dimension'])

        self.encoder_projector = nn.Sequential(
            self.encoder,
            self.projector
        )

        self.predictor = prediction_MLP(config['model']['feat_dim'])

    def forward(self, x):
        
        im_aug1 = x[:, 0]
        im_aug2 = x[:, 1]
       # im_aug1 = maybePermuteInput(im_aug1, self.config)
      #  im_aug2 = maybePermuteInput(im_aug2, self.config)
        z1 = self.encoder_projector(im_aug1)
        z2 = self.encoder_projector(im_aug2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        outputs = {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}
        return outputs
# GRAVEYARD #
# class EncoderDecoderModel(EncoderModel):
#     def __init__(self, in_channels, backbone_name, num_classes):
#         super(EncoderDecoderModel, self).__init__(in_channels, backbone_name)
#         self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
#         self.decoder = nn.Sequential(*list(self.encoder.children())[:-1])

#     def forward(self, x):
#         z = self.encoder(x)
#         p = self.decoder(z)
#         outputs = {'p1': p, 'z1': z}
#         return outputs


# class EncoderDecoderSkipsModel(EncoderModel):
#     def __init__(self, in_channels, backbone_name, num_classes):
#         super(EncoderDecoderModel, self).__init__(in_channels, backbone_name)
#         self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

#     def forward(self, x):
#         skips = []

#         features = nn.Sequential(*list(self.encoder.children())[
#             :self.skip_layers[0]])(x)
#         skips.append(features)

#         # Extract intermediate representations
#         for i in range(self.skip_layers[0], self.skip_layers[1]):
#             features = nn.Sequential(
#                 *list(self.encoder.children())[i:i+1])(features)

#             skips.append(features)

#         outputs = skips
#         return outputs
