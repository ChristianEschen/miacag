from torch import nn
from models.mlps import prediction_MLP, projection_MLP
from segmentation_models_pytorch.encoders import get_encoder


class EncoderModel(nn.Module):
    def __init__(self, in_channels, backbone_name, depth=5,
                 ):
        super(EncoderModel, self).__init__()
        self.encoder = get_encoder(backbone_name,
                                   in_channels=in_channels,
                                   depth=depth)

    def forward(self, x):
        z = self.encoder(x)
        return z


class ClassificationModel(EncoderModel):
    def __init__(self, in_channels, backbone_name, num_classes, depth=5,
                 dimension='2D+T'):
        super(ClassificationModel, self).__init__(
            in_channels, backbone_name, depth)
        self.fc = nn.Linear(self.encoder.out_channels[-1], num_classes)
        self.dimension = dimension

    def forward(self, x):
        p = self.encoder(x)
        p = p[-1]
        if self.dimension in ['3D', '2D+T']:
            p = p.mean(dim=(-3, -2, -1))
        else:
            p = p.mean(dim=(-2, -1))
        p = p.view(-1, self.encoder.out_channels[-1])

        p = self.fc(p)
        return p


class SimSiam(EncoderModel):
    def __init__(self, in_channels, backbone_name, feat_dim,
                 num_proj_layers, depth=5, dimension='2D+T'):
        super(SimSiam, self).__init__(
            in_channels, backbone_name, depth=depth)
        self.projector = projection_MLP(self.encoder.out_channels[-1],
                                        feat_dim,
                                        num_proj_layers,
                                        dimension)

        self.encoder_projector = nn.Sequential(
            self.encoder,
            self.projector
        )

        self.predictor = prediction_MLP(feat_dim)

    def forward(self, x):
        im_aug1 = x[:, 0]
        im_aug2 = x[:, 1]
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
