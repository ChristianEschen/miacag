from torch import nn
from models.backbone_encoders._2D.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.mlps import prediction_MLP, projection_MLP


class EncoderModel(nn.Module):
    def __init__(self, in_channels, backbone_name):
        super(EncoderModel, self).__init__()
        self.encoder = self.get_backbone(backbone_name, in_channels)

    def get_backbone(self, backbone_name, in_channels):
        return {'resnet18': ResNet18(in_channels),
                'resnet34': ResNet34(in_channels),
                'resnet50': ResNet50(in_channels),
                'resnet101': ResNet101(in_channels),
                'resnet152': ResNet152(in_channels)}[backbone_name]

    def forward(self, x):
        z = self.encoder(x)
        outputs = {'z1', z}
        return outputs


class ClassificationModel(EncoderModel):
    def __init__(self, in_channels, backbone_name, num_classes):
        super(ClassificationModel, self).__init__(in_channels, backbone_name)
        self.out_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Linear(self.out_dim, num_classes)

    def forward(self, x):
        p = self.encoder(x)
        outputs = {'p1', p}
        return outputs


class SimSiam(EncoderModel):
    def __init__(self, in_channels, backbone_name, feat_dim, num_proj_layers):
        super(SimSiam, self).__init__(in_channels, backbone_name)
        out_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

        self.projector = projection_MLP(out_dim, feat_dim,
                                        num_proj_layers)

        self.encoder = nn.Sequential(
            self.encoder,
            self.projector
        )

        self.predictor = prediction_MLP(feat_dim)

    def forward(self, x):
        im_aug1 = x[:, 0]
        im_aug2 = x[:, 1]
        z1 = self.encoder(im_aug1)
        z2 = self.encoder(im_aug2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        outputs = {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}
        return outputs
