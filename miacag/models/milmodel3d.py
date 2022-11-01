from typing import Dict, Optional, Union, cast
from miacag.models.modules import EncoderModel, ImageToScalarModel
import torch
import torch.nn as nn
from miacag.models.get_encoder import get_encoder, modelsRequiredPermute
from miacag.models.modules import maybePermuteInput, get_final_layer
from monai.utils.module import optional_import
from miacag.models.modules import unique_counts

models, _ = optional_import("torchvision.models")


class MILModel(ImageToScalarModel):
    def __init__(
        self,
        config,
        device
    ) -> None:
        super().__init__(config, device)
        self.config = config
        if all(i <= 0 for i in self.config['model']['num_classes']):
            raise ValueError(
                "Number of classes must be positive: "
                + str(self.config['model']['num_classes']))

        if self.config['model']['mil_mode'].lower() not in [
                "mean", "max", "att", "att_trans", "att_trans_pyramid"]:
            raise ValueError(
                "Unsupported mil_mode: "
                + str(self.config['model']['mil_mode']))

        self.mil_mode = self.config['model']['mil_mode'].lower()
        self.attention = nn.Sequential()
        self.transformer = None  # type: Optional[nn.Module]
        self.loss_uniques, self.count_loss_unique = unique_counts(self.config)
        # define mil_mode
        if self.mil_mode in ["mean", "max"]:
            pass
        elif self.mil_mode == "att":
            self.attention = nn.ModuleList()
            for c, head in enumerate(self.loss_uniques):
                self.attention.append(nn.Sequential(
                    nn.Linear(self.in_features, 2048),
                    nn.Tanh(),
                    nn.Linear(2048, 1)))

        elif self.mil_mode == "att_trans":
            self.attention = nn.ModuleList()
            self.transformer = nn.ModuleList()
            for c, head in enumerate(self.loss_uniques):
                transformer = nn.TransformerEncoderLayer(
                    d_model=self.in_features, nhead=8,
                    dropout=self.config['model']['trans_dropout'])
                self.transformer.append(nn.TransformerEncoder(
                    transformer,
                    num_layers=self.config['model']['trans_blocks']))
                self.attention.append(nn.Sequential(
                    nn.Linear(self.in_features, 2048),
                    nn.Tanh(),
                    nn.Linear(2048, 1)))

        elif self.mil_mode == "att_trans_pyramid":
            self.attention = nn.ModuleList()
            self.transformer = nn.ModuleList()
            for c, head in enumerate(self.loss_uniques):
                transformer_list = nn.ModuleList(
                    [
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(
                                d_model=256,
                                nhead=8,
                                dropout=self.config['model']['trans_dropout']),
                            num_layers=self.config['model']['trans_blocks']
                        ),
                        nn.Sequential(
                            nn.Linear(768, 256),
                            nn.TransformerEncoder(
                                nn.TransformerEncoderLayer(
                                    d_model=256, nhead=8,
                                    dropout=self.config['model']['trans_dropout']),
                                num_layers=self.config['model']['trans_blocks'],
                            ),
                        ),
                        nn.Sequential(
                            nn.Linear(1280, 256),
                            nn.TransformerEncoder(
                                nn.TransformerEncoderLayer(
                                    d_model=256, nhead=8,
                                    dropout=self.config['model']['trans_dropout']),
                                num_layers=self.config['model']['trans_blocks'],
                            ),
                        ),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(
                                d_model=2304, nhead=8,
                                dropout=self.config['model']['trans_dropout']),
                            num_layers=self.config['model']['trans_blocks'],
                        ),
                    ]
                )
                self.transformer.append(transformer_list)
                self.in_features = self.in_features + 256
                self.attention.append(nn.Sequential(
                    nn.Linear(self.in_features, 2048),
                    nn.Tanh(),
                    nn.Linear(2048, 1)))

        else:
            raise ValueError("Unsupported mil_mode: " + str(self.mil_mode))




    def calc_head(self, x: torch.Tensor, c: int) -> torch.Tensor:

        sh = x.shape

        if self.mil_mode == "mean":
            #x = self.fcs_func(self.fcs, c)(x)
            x = self.fcs[0](x)
            x = torch.mean(x, dim=1)
            a = None
        elif self.mil_mode == "max":
            x = self.fcs[0](x)
          #  x = self.fcs_func(self.fcs, c)(x)
            x, _ = torch.max(x, dim=1)
            a = None
        elif self.mil_mode == "att":
            a = self.attention[0](x)
            #a = self.attention_func(self.attention, c)(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            x = self.fcs[0](x)
            #x = self.fcs_func(self.fcs, c)(x)

        elif self.mil_mode == "att_trans" and self.transformer is not None:

            x = x.permute(1, 0, 2)
            a = self.attention[0](x)
         #   x = self.transform_func(self.transformer, c)(x)

            x = x.permute(1, 0, 2)
            a = self.attention[0](x)
           # a = self.attention_func(self.attention, c)(x)

            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            #x = self.fcs_func(self.fcs, c)(x)
            x = self.fcs[0](x)

        elif self.mil_mode == "att_trans_pyramid" \
                and self.transformer is not None:

            l1 = torch.mean(
                self.extra_outputs["layer1"], dim=(2, 3)).reshape(
                    sh[0], sh[1], -1).permute(1, 0, 2)
            l2 = torch.mean(
                self.extra_outputs["layer2"], dim=(2, 3)).reshape(
                    sh[0], sh[1], -1).permute(1, 0, 2)
            l3 = torch.mean(
                self.extra_outputs["layer3"], dim=(2, 3)).reshape(
                    sh[0], sh[1], -1).permute(1, 0, 2)
            l4 = torch.mean(
                self.extra_outputs["layer4"], dim=(2, 3)).reshape(
                    sh[0], sh[1], -1).permute(1, 0, 2)

            transformer_list = cast(nn.ModuleList, self.transformer[c])

            transformer_list = cast(nn.ModuleList, self.transformer[c])

            x = transformer_list[0](l1)
            x = transformer_list[1](torch.cat((x, l2), dim=2))
            x = transformer_list[2](torch.cat((x, l3), dim=2))
            x = transformer_list[3](torch.cat((x, l4), dim=2))

            x = x.permute(1, 0, 2)

            #a = self.attention_func(self.attention, c)(x)
            a = self.attention[0](x)

            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            #x = self.fcs_func(self.fcs, c)(x)
            x = self.fcs[0](x)

        else:
            raise ValueError("Wrong model mode" + str(self.mil_mode))

        return x, a

    def forward(self, x: torch.Tensor, no_head: bool = False) -> torch.Tensor:

        sh = x.shape

        x = self.handle_x_dim_input(x, sh)
        x = maybePermuteInput(x, self.config)
        x = self.encoder(x)
        x = self.reduce_feature_space(x)

        x = x.reshape(sh[0], sh[1], -1)

        xs = []
        if not no_head:
            for c in range(0, len(self.fcs)):
                x_c, _ = self.calc_head(x, c)
                xs.append(x_c)
           # xs = self.calc_head(x)
        else:
            xs = x
        return xs

    def get_attention(self, x: torch.Tensor, no_head: bool = False):
      #  self.config['loaders']['val_method']['saliency'] = 'False'
        sh = x.shape
        x = self.handle_x_dim_input(x, sh)
        x = maybePermuteInput(x, self.config)
        x = self.encoder(x)
        x = self.reduce_feature_space(x)
        x = x.reshape(sh[0], sh[1], -1)
        a_s = []
        xs = []
        if not no_head:
            for c in range(0, len(self.fcs)):
                x_c, a_c = self.calc_head(x, c)
                xs.append(x_c)
                a_s.append(a_c)
        else:
            xs = x
        self.config['loaders']['val_method']['saliency'] = 'True'
        return xs, a_c

    # hacks for saliency maps
    # while running with saliency maps enabled,
    # we change the forward method for the model
    # to use the forward_saliency method
    def transform_func(self, transformer, c):
        if (self.config['loaders']['val_method']['saliency'] == 'True'
                and
                self.config['loaders']['mode'] == 'prediction'):
            transformer = transformer
        else:
            transformer = transformer[c]
        return transformer

    def fcs_func(self, fcs, c):
        if (self.config['loaders']['val_method']['saliency'] == 'True'
                and
                self.config['loaders']['mode'] == 'prediction'):
            fcs = fcs
        else:
            fcs = fcs[c]
        return fcs

    def attention_func(self, attention, c):
        if (self.config['loaders']['val_method']['saliency'] == 'True'
                and
                self.config['loaders']['mode'] == 'prediction'):
            attention = attention[c]
        else:
            attention = attention[c]
        return attention

    def forward_saliency(self, x):
        sh = x.shape

        #x = self.handle_x_dim_input(x, sh)
        x = maybePermuteInput(x, self.config)
        p = self.encoder(x)
        p = self.reduce_feature_space(p)

        x = self.fcs[0](p) 
        # x = maybePermuteInput(x, self.config)
        # p = self.encoder(x)
        # if self.dimension in ['3D', '2D+T']:
        #     if self.config['model']['backbone'] not in ["mvit_base_16x4", "mvit_base_32x3"]:
        #         p = p.mean(dim=(-3, -2, -1))
        #     else:
        #         pass
        # else:
        #     raise ValueError(
        #         'not implemented for dimension: %s' % self.config['model'])
        # x = self.fcs[0](p) # p = encoder
        return p

    def handle_x_dim_input(self, x: torch.Tensor, sh: tuple):
        if self.config['model']['dimension'] == '2D':
            x = self.get_forward_2d(x, sh)
        elif self.config['model']['dimension'] == '2D+T':
            x = self.get_forward_3d(x, sh)
        else:
            raise ValueError(
                'not implemented for dimension: %s' %
                self.config['model']['dimension'])
        return x

    def get_forward_2d(self, x: torch.Tensor, sh: tuple):
        x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])

        return x
    
    def get_forward_3d(self, x: torch.Tensor, sh: tuple):
        x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4], sh[5])
        return x

    def reduce_feature_space(self, x):
        if self.config['model']['dimension'] in ['3D', '2D+T']:
            if self.config['model']['backbone'] not in ["mvit_base_16x4", "mvit_base_32x3"]:
                x = x.mean(dim=(-3, -2, -1))
            else:
                pass
        elif self.config['model']['dimension'] in ['2D']:
            x = x.mean(dim=(-2, -1))
        else:
            raise ValueError('this dimension is not implemented')
        return x
