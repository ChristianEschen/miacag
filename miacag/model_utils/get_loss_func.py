import torch.nn as nn
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from miacag.model_utils.siam_loss import SimSiamLoss
from miacag.models.modules import unique_counts
import torch
from torch import Tensor
import copy
import numpy as np

import torch.nn.functional as F

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
    sum_e = events.sum()+eps
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(sum_e)


def l1_loss_smooth_sten2(predictions, targets, beta=1):
    beta_l1, phase, *lambda_weights = beta
    lambda_weights = torch.tensor(lambda_weights, device=predictions.device)

    if predictions.shape[0] == 0:
        return torch.tensor(0.0, device=predictions.device)

    loss = 0.0
    for segment_idx in range(targets.size(1)):
        y = targets[:, segment_idx]
        x = predictions[:, segment_idx]

        # Adjust target values during training phase
        if phase == 1:
            y[y < 0.01] = (0 - 0.5) * torch.rand_like(y[y < 0.01]) + 0.5

        # Compute the Huber loss components
        abs_diff = torch.abs(x - y)
        quadratic = torch.where(abs_diff < beta_l1, 0.5 * (x - y) ** 2 / beta_l1, abs_diff - 0.5 * beta_l1)

        # Apply lambda weights conditionally
        lambda_mask = (y >= 0.01).float()
        weighted_quadratic = quadratic * (1 + lambda_mask * (lambda_weights[segment_idx] - 1))
        
        loss += weighted_quadratic.mean()

    loss /= targets.size(1)
    return loss

def l1_loss_smooth_sten(predictions, targets, beta=1):
    beta_l1 = beta[0]
    # copy beta to lampda
    phase = beta[1] # phase=1 means train and phase=0 means val
    lambda_weights = beta.copy()
    lambda_weights = lambda_weights[2:]
    # cast lambda weights to gpu
    lambda_weights = torch.tensor(lambda_weights, device=predictions.device)
  #  mask = torch.isnan(targets)
    loss = 0
    #predictions = predictions[~mask]
    # predictions = predictions.masked_select(~mask)
    # targets = targets[~mask]
    # lambda_weights = lambda_weights.masked_select(~mask)
   # print('lambda_weights', lambda_weights)
   # print('predictions', predictions)
    
    if predictions.shape[0] != 0:
        for segment_idx in range(0, targets[0,:].shape[0]):
            #or x, y in zip(predictions, targets):
            y = targets[:,segment_idx]
            x = predictions[:,segment_idx]
            c = 0
            for y_i in y:
                x_i = x[c]
                y_i_orig = copy.deepcopy(y_i)
                if phase == 1:
                    if y_i < 0.01:
                        y_i = (0 - 0.5) * torch.rand(1, device=predictions.device) + 0.5
                if abs(x_i-y_i) < beta_l1:
                    if y_i_orig >= 0.01:
                        loss += (0.5*(x_i-y_i)**2 / beta_l1).mean()*lambda_weights[segment_idx]
                    else:
                        
                        loss += (0.5*(x_i-y_i)**2 / beta_l1).mean()
                else:
                    if y_i_orig >= 0.01:
                        loss += (abs(x_i-y_i) - 0.5 * beta_l1).mean()*lambda_weights[segment_idx]
                    else:
                        loss += (abs(x_i-y_i) - 0.5 * beta_l1).mean()
                c += 1
        loss = loss/predictions.shape[0]
        return loss
    else:
        loss = torch.tensor(0.0, device=predictions.device)
        return loss
    
def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def l1_loss_smooth(predictions, targets, beta=1):
    mask = torch.isnan(targets)
    loss = 0
    #predictions = predictions[~mask]
    predictions = predictions.masked_select(~mask)
    targets = targets[~mask]
    c = 0
    if predictions.shape[0] != 0:
        for x, y in zip(predictions, targets):
            if abs(x-y) < beta:
                loss += (0.5*(x-y)**2 / beta).mean()
            else:
                loss += (abs(x-y) - 0.5 * beta).mean()
            c += 1
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

def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_l1_loss_multioutput(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = 0
   # torch.tensor(0.0, device=predictions.device)
    for i in range(0, inputs.shape[1]):

        loss += weighted_focal_l1_loss(inputs[:,i],targets[:,i], weights=weights[:,i])
    return loss
        
def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    mask = torch.isnan(targets)
    targets = targets[~mask]
    inputs = inputs[~mask]
    weights = weights[~mask]

    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    if targets.size()==torch.Size([0]):
        loss=torch.tensor(0.0, device=inputs.device, requires_grad=True)
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
            
            # if config["labels_names"][0].startswith('sten'):
            #     criterion = l1_loss_smooth_sten
            #     l1_loss_smooth_sten.__defaults__=([config['loss']['beta']] +[1]+ config['loss']['lambda_weights'],)
            #     # criterion = l1_loss_smooth_sten2
            #     # l1_loss_smooth_sten2.__defaults__=([config['loss']['beta']] +[1]+ config['loss']['lambda_weights'],)
            # else:
            #     criterion = l1_loss_smooth
            #     l1_loss_smooth.__defaults__=(config['loss']['beta'],)
            # criterion = weighted_huber_loss
            criterion = weighted_huber_loss
            weighted_huber_loss.__defaults__=(config['loss']['beta'],)
           # weighted_huber_loss(inputs, targets, weights=None, beta=1.)
            criterions.append(criterion)
        elif loss.startswith('wfocall1'):
            criterion = weighted_focal_l1_loss_multioutput
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
