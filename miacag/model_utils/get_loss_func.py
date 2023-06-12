import torch.nn as nn
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from miacag.model_utils.siam_loss import SimSiamLoss
from miacag.models.modules import unique_counts
import torch
from torch import Tensor


def mse_loss_with_nans(input, target):

    # Missing data are nan's
    mask = torch.isnan(target)

    # Missing data are 0's
   # mask = target == 99998

    out = (input[~mask]-target[~mask])**2
    loss = out.mean()

    return loss

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    durations = durations.view(-1)
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    # log_h = log_h[idx] (deprecated)
    log_h = log_h.index_select(0, idx)
    return cox_ph_loss_sorted(log_h, events, eps)


def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())



def l1_loss_smooth(predictions, targets, beta=1):
    mask = torch.isnan(targets)
    loss = 0
    #predictions = predictions[~mask]
    predictions = predictions.masked_select(~mask)
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
        elif loss.startswith('NNL'):
            criterion = cox_ph_loss
            criterions.append(criterion)
        else:
            raise ValueError("Loss type is not implemented")
    return criterions
