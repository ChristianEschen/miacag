import torch.nn as nn
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from miacag.model_utils.siam_loss import SimSiamLoss
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


def get_loss_func(config):
    criterions = []
    for loss in config['loss']['name']:
        if loss == 'CE':
            criterion = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=99998)
            criterions.append(criterion)
        elif loss == 'MSE':

            #criterion = torch.nn.MSELoss(reduce=True, reduction='mean')
            criterion = mse_loss_with_nans  # (input, target)
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
