from enum import unique
from torch import nn
from miacag.models.mlps import prediction_MLP, projection_MLP
from miacag.models.get_encoder import get_encoder, modelsRequiredPermute
import torch.nn.functional as F
import torch
import numpy as np


def getCountsLoss(losses):
    count_regression = 0
    count_classification = 0
    for loss_u in losses:
        if loss_u in ['CE']:
            count_classification += 1
        elif loss_u in ['MSE', 'L1', 'L1smooth']:
            count_regression += 1
        else:
            raise ValueError('this loss is not implementeed:', loss_u)
    return count_regression, count_classification


def get_num_class_for_loss_group(losses, loss_group, num_classes):
    for loss in range(0, len(losses)):
        if losses[loss] == loss_group:
            return num_classes[loss]


def unique_counts(config, remove_total=False):
    if remove_total:
        if 'total' in config['loss']['name']:
            config['loss']['name'].remove('total')
    loss_uniques, count = np.unique(np.array(
        config['loss']['name']), return_counts=True)
    count = count.tolist()
    loss_uniques = loss_uniques.tolist()
    return loss_uniques, count


def maybePermuteInput(x, config):
    model_list = modelsRequiredPermute()
    if config['model']['backbone'] in model_list:
        x = x.permute(0, 1, 4, 2, 3)
        return x
    else:
        return x


def maybePackPathway(frames, config):
    if config['model']['backbone'] == 'slowfast8x8':
        fast_pathway = frames
        alpha = 4
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long())
        frame_list = [slow_pathway, fast_pathway]
        return frame_list
    else:
        return frames


def get_final_layer(config,  device, in_features):
    fcs = []
    c = 0
    for head in range(0, len(config['labels_names'])):
        if config['loss']['name'][c] in ['CE']:
            fcs.append(
                nn.Linear(
                    in_features,
                    config['model']['num_classes']).to(device))
        elif config['loss']['name'][c] in ['MSE', 'L1']:
            fcs.append(
                nn.Sequential(
                    nn.Linear(
                        in_features, 1).to(device),
                    nn.ReLU()))
        else:
            raise ValueError('loss not implemented')
        c += 1
    return fcs


class EncoderModel(nn.Module):
    def __init__(self, config, device):
        super(EncoderModel, self).__init__()

        self.encoder, self.in_features = get_encoder(config, device)
        
    def forward(self, x):
        x = maybePermuteInput(x, self.config)
        z = self.encoder(x)
        return z


class ImageToScalarModel(EncoderModel):
    def __init__(self, config, device):
        super(ImageToScalarModel, self).__init__(config, device)
        self.config = config
       # self.fcs = ["self.fc" ]
        self.fcs = nn.ModuleList()
        c = 0
       # uniques = 
        loss_uniques, counts = unique_counts(self.config)

        for loss_count_idx, loss_type in enumerate(loss_uniques):
            count_loss = counts[loss_count_idx]
            if count_loss > 0:
                num_classes = get_num_class_for_loss_group(
                    self.config['loss']['name'],
                    loss_type, self.config['model']['num_classes'])
                if loss_type.startswith('CE'):
                    self.fcs.append(nn.Linear(
                            self.in_features,
                            num_classes).to(device))
                elif loss_type in ['BCE_multilabel']:
                    self.fcs.append(
                        nn.Sequential(
                            nn.Linear(
                                self.in_features, count_loss)
                            ).to(device))
                elif loss_type in ['MSE', 'L1', 'L1smooth']:
                    self.fcs.append(
                        nn.Sequential(
                            nn.Linear(
                                self.in_features,
                                count_loss).to(device)
                        ))
                else:
                    raise ValueError('loss not implemented')
            c += 1
        self.dimension = config['model']['dimension']

    def forward(self, x):
        x = maybePermuteInput(x, self.config)
        p = self.encoder(x)
        if self.dimension in ['3D', '2D+T']:
            if self.config['model']['backbone'] not in ["mvit_base_16x4", "mvit_base_32x3"]:
                p = p.mean(dim=(-3, -2, -1))
            else:
                pass
        elif self.dimension == 'tabular':
            p = p
        else:
            p = p.mean(dim=(-2, -1))
        ps = []
        for fc in self.fcs:
            ps.append(fc(p))
        return ps

    def forward_saliency(self, x):
        x = maybePermuteInput(x, self.config)
        p = self.encoder(x)
        if self.dimension in ['3D', '2D+T']:
            if self.config['model']['backbone'] not in ["mvit_base_16x4", "mvit_base_32x3"]:
               p = p.mean(dim=(-3, -2, -1))
               # p = p[:, 1:, :].reshape(p.size(0), 8, 7, 7, p.size(2))
            else:
                pass
        elif self.dimension == 'tabular':
            p = p
        else:
            p = p.mean(dim=(-2, -1))
        ps = self.fcs[0](p)
        return ps
    


class SimSiam(EncoderModel):
    def __init__(self, config, device):
        super(SimSiam, self).__init__(
            config, device)
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
