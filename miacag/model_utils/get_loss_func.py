import torch.nn as nn
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from miacag.model_utils.siam_loss import SimSiamLoss
from miacag.models.modules import unique_counts
import torch


def mse_loss_with_nans(input, target):

    # Missing data are nan's
    mask = torch.isnan(target)

    # Missing data are 0's
   # mask = target == 99998

    out = (input[~mask]-target[~mask])**2
    loss = out.mean()

    return loss


# def l1_loss_smooth(predictions, targets, beta=1):
#     # Compute the absolute difference between predictions and targets
#     diff = torch.abs(predictions - targets)

#     # Compute the mask for missing values in the targets
#     mask = torch.isnan(targets)

#     # Replace the missing values in the mask with the value of beta
#     diff = torch.where(mask, beta, diff)

#     # Compute the loss as the mean of the smooth L1 loss over all samples
#     return diff.mean()

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
    for loss in config['loss']['groups_names']:
        if loss.startswith('CE'):
            criterion = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=99998)
            criterions.append(criterion)
        elif loss.startswith('BCE_multilabel'):
            criterion = bce_with_nans
            criterions.append(criterion)
        elif loss.startswith('MSE'):

            #criterion = torch.nn.MSELoss(reduce=True, reduction='mean')
            criterion = mse_loss_with_nans  # (input, target)
            criterions.append(criterion)
        elif loss.startswith('_L1'):
            criterion = mae_loss_with_nans  # (input, target)
            criterions.append(criterion)
        elif loss.startswith('L1smooth'):
            criterion = l1_loss_smooth
            l1_loss_smooth.__defaults__=(config['loss']['beta'],)
            criterions.append(criterion)
        elif loss.startswith('dice_loss'):
            criterion = DiceLoss(
                include_background=False,
                to_onehot_y=False, sigmoid=False,
                softmax=True, squared_pred=True)
            criterions.append(criterion)
        elif loss.startswith('diceCE_loss'):
            criterion = DiceCELoss(
                include_background=True,
                to_onehot_y=False, sigmoid=False,
                softmax=True, squared_pred=True)
            criterions.append(criterion)
        elif loss.startswith('Siam'):
            criterion = SimSiamLoss('original')
            criterions.append(criterion)
        elif loss.startswith('total'):
            pass
        else:
            raise ValueError("Loss type is not implemented")
    return criterions
