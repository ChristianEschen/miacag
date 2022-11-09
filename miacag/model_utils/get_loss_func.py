import torch.nn as nn
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from miacag.model_utils.siam_loss import SimSiamLoss
from miacag.models.modules import unique_counts
import torch


# def mse_loss(self, input, target, ignored_index, reduction):
#     mask = target == ignored_index
#     out = (input[~mask]-target[~mask])**2
#     if reduction == "mean":
#         return out.mean()
#     elif reduction == "None":
#         return out
def mse_loss_with_nans(input, target):

    # Missing data are nan's
    mask = torch.isnan(target)

    # Missing data are 0's
   # mask = target == 99998

    out = (input[~mask]-target[~mask])**2
    loss = out.mean()

    return loss


def l1_loss_smooth(predictions, targets, beta=1):
    mask = torch.isnan(targets)
    loss = 0
    predictions = predictions[~mask]
    targets = targets[~mask]
    if predictions.shape[0] != 0:
        for x, y in zip(predictions, targets):
            if abs(x-y) < beta:
                loss += (0.5*(x-y)**2 / beta).mean()
            else:
                loss += (abs(x-y) - 0.5 * beta).mean()
        loss = loss/predictions.shape[0]
        return loss
    else:
        loss = torch.tensor(0.0, device=predictions.device)
        return loss


def bce_with_nans(predictions, targets):
    mask = torch.isnan(targets)
    loss = 0
    predictions = predictions[~mask]
    targets = targets[~mask]
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    loss = criterion(predictions, targets.float())
    # for x, y in zip(predictions, targets):
        
    #     if abs(x-y) < beta:
    #         loss += (0.5*(x-y)**2 / beta).mean()
    #     else:
    #         loss += (abs(x-y) - 0.5 * beta).mean()

    # loss = loss/predictions.shape[0]
    return loss


def mae_loss_with_nans(input, target):

    # Missing data are nan's
    mask = torch.isnan(target)

    # Missing data are 0's
   # mask = target == 99998

    out = torch.abs(input[~mask]-target[~mask])
    loss = out.mean()

    return loss


def get_loss_func(config):
    criterions = []
    loss_names, loss_name_counts = unique_counts(config)
    for loss in loss_names:
        if loss.startswith('CE'):
            criterion = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=99998)
            criterions.append(criterion)
        elif loss == 'BCE_multilabel':
            criterion = bce_with_nans
            criterions.append(criterion)
        elif loss == 'MSE':

            #criterion = torch.nn.MSELoss(reduce=True, reduction='mean')
            criterion = mse_loss_with_nans  # (input, target)
            criterions.append(criterion)
        elif loss == 'L1':
            criterion = mae_loss_with_nans  # (input, target)
            criterions.append(criterion)
        elif loss == 'L1smooth':
            criterion = l1_loss_smooth
            l1_loss_smooth.__defaults__=(config['loss']['beta'],)
            criterions.append(criterion)
        elif loss == 'dice_loss':
            criterion = DiceLoss(
                include_background=False,
                to_onehot_y=False, sigmoid=False,
                softmax=True, squared_pred=True)
            criterions.append(criterion)
        elif loss == 'diceCE_loss':
            criterion = DiceCELoss(
                include_background=True,
                to_onehot_y=False, sigmoid=False,
                softmax=True, squared_pred=True)
            criterions.append(criterion)
        elif loss == 'Siam':
            criterion = SimSiamLoss('original')
            criterions.append(criterion)
        elif loss == 'total':
            pass
        else:
            raise ValueError("Loss type is not implemented")
    return criterions
