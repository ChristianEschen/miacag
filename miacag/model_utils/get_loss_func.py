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

# class NLLSurvLoss(nn.Module):
#     """
#     The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
#     Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
#     Parameters
#     ----------
#     alpha: float
#         TODO: document
#     eps: float
#         Numerical constant; lower bound to avoid taking logs of tiny numbers.
#     reduction: str
#         Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
#     """
#     def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
#         super().__init__()
#         self.alpha = alpha
#         self.eps = eps
#         self.reduction = reduction

#     def __call__(self, h, y, c):
#         """
#         Parameters
#         ----------
#         h: (n_batches, n_classes)
#             The neural network output discrete survival predictions such that hazards = sigmoid(h).
#         y_c: (n_batches, 2) or (n_batches, 3)
#             The true time bin label (first column) and censorship indicator (second column).
#         """

#         return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
#                         alpha=self.alpha, eps=self.eps,
#                         reduction=self.reduction)


# # TODO: document better and clean up
# def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
#     """
#     The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
#     Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
#     Parameters
#     ----------
#     h: (n_batches, n_classes)
#         The neural network output discrete survival predictions such that hazards = sigmoid(h).
#     y: (n_batches, 1)
#         The true time bin index label.
#     c: (n_batches, 1)
#         The censoring status indicator.
#     alpha: float
#         The weight on uncensored loss 
#     eps: float
#         Numerical constant; lower bound to avoid taking logs of tiny numbers.
#     reduction: str
#         Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
#     References
#     ----------
#     Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
#     """
#     # print("h shape", h.shape)

#     # make sure these are ints
#    # device = "cuda:{}".format(os.environ['LOCAL_RANK'])
#     with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):


#         y = y.type(torch.int64)
#         # remove last singleton dimension
#         y = y.squeeze(dim=1)
#         c = c.type(torch.int64)

#         hazards = torch.sigmoid(h)
#         # print("hazards shape", hazards.shape)

#         S = torch.cumprod(1 - hazards, dim=1)
#         # print("S.shape", S.shape, S)

#         S_padded = torch.cat([torch.ones_like(c), S], 1)
#         # S(-1) = 0, all patients are alive from (-inf, 0) by definition
#         # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
#         # hazards[y] = hazards(1)
#         # S[1] = S(1)
#         # TODO: document and check

#         # print("S_padded.shape", S_padded.shape, S_padded)


#         # TODO: document/better naming
#         s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
#         h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
#         s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
#         # print('s_prev.s_prev', s_prev.shape, s_prev)
#         # print('h_this.shape', h_this.shape, h_this)
#         # print('s_this.shape', s_this.shape, s_this)

#         # c = 1 means censored. Weight 0 in this case 
#         uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
#         censored_loss = - c * torch.log(s_this)
        

#         # print('uncensored_loss.shape', uncensored_loss.shape)
#         # print('censored_loss.shape', censored_loss.shape)

#         neg_l = censored_loss + uncensored_loss
#         if alpha is not None:
#             loss = (1 - alpha) * neg_l + alpha * uncensored_loss

#         if reduction == 'mean':
#             loss = loss.mean()
#         elif reduction == 'sum':
#             loss = loss.sum()
#         else:
#             raise ValueError("Bad input for reduction: {}".format(reduction))

#     return loss


import torch
import torch.nn as nn

class NLLSurvLoss(nn.Module):
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(self, h, y, c):
        with torch.cuda.amp.autocast(enabled=False):
            return nll_loss(h=h.float(), y=y.unsqueeze(dim=1).float(), c=c.unsqueeze(dim=1).float(),
                            alpha=self.alpha, eps=self.eps,
                            reduction=self.reduction)

def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    y = y.to(dtype=torch.int64).squeeze(dim=1)
    c = c.to(dtype=torch.int64)

    hazards = torch.sigmoid(h)
    S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c, dtype=h.dtype), S], 1)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=eps)

    uncensored_loss = -(1 - c.to(dtype=h.dtype)) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = -c.to(dtype=h.dtype) * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss



def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

def nll_logistic_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                        reduction: str = 'mean') -> Tensor:
    """Negative log-likelihood of the discrete time hazard parametrized model LogisticHazard [1].
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    # cast idx_durations to int
    idx_durations = idx_durations.long()
    events = events.float()
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    events = events.to(phi.dtype)
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)



def gradient_blending_nnl_loss(outputs, labels, events):
    outputs_total = outputs[0]
    outputs_tab = outputs[1]
    outputs_vis = outputs[2]

    tv_loss = nll_logistic_hazard(outputs_total, labels, events)
    t_loss = nll_logistic_hazard(outputs_tab, labels, events)
    v_loss = nll_logistic_hazard(outputs_vis, labels, events)
    
    weighted_t_loss = t_loss * 1
    weighted_v_loss = v_loss * 1 
    weighted_tv_loss = tv_loss * 1
    
    loss = weighted_t_loss + weighted_v_loss + weighted_tv_loss
    return loss
    
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


def bce_with_nans(predictions, targets, config=None, train=True):
    # mask = torch.isnan(targets)
    # loss = 0
    # predictions = predictions[~mask]

    # targets = targets[~mask]
    pos_weights = []
    for lab_name in config['labels_names']:
        pos_weights.append(config['labels_' + lab_name + '_weights'])
    pos_weights = torch.concatenate(pos_weights).to(predictions.device)
    # cat pos_weights to the same dtype as predictions
    pos_weights = pos_weights.float()
    if train:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='mean')
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    predictions = predictions.float()
    targets = targets.float()   

    loss = criterion(predictions, targets)
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

def find_matches_two_lists(labels_names, indentify_cor_type):
    match = []
    for i in indentify_cor_type:
        count = 0
        for label_name in labels_names:
            
            if i in label_name:
                match.append(count)
            count += 1
            
    return match

def create_ll(liste, length):
    ll = []
    for i in range(0, length):
        ll.append(liste)
    return ll

def create_tensor_from_list(ll, rows, cols):
   # num_rows = len(ll)
   # num_cols = max(max(sublist) for sublist in ll) + 1  # Find the max index and adjust for base-0 indexing

    # Initialize a tensor of shape (num_rows, num_cols) filled with zeros (False)
    mask_tensor = torch.zeros((rows, cols), dtype=torch.bool)

    # Fill the tensor with True where indices are specified
    for row_idx, sublist in enumerate(ll):
        for col_idx in sublist:
            mask_tensor[row_idx, col_idx] = True

        # Convert boolean tensor to float
    float_tensor = mask_tensor.float()  # Converts True to 1.0 and False to 0.0

    # Replace 0.0 with NaN
    float_tensor[float_tensor == 0] = float('nan')

    return float_tensor


def create_nan_tensor_to_remove_segments_from_targets(labels_names, length, identify_cor_type):
    
    ll = create_ll(labels_names, length)
    idx_ll = find_matches_ll(ll, identify_cor_type)
    float_tensor = create_tensor_from_list(idx_ll, length, len(labels_names))
    return float_tensor


def find_matches_ll(labels_names_ll, indentify_cor_type):
    identify_cor_artery_rca = ['_1_prox',  '_2_mi' , '_3_dist', '_4_pda_trans', '_16_pla_rca']
    identify_cor_artery_lca = [ '_4_pda_lca', '_5_lm', '_6_prox', '_7_mi', '_8_dist',
                            '_9_d1', '_10_d2', '_11_prox', '_12_om', '_13_midt', '_14_om', '_15_dist', '_16_pla_lca']
    #indentify_cor_type = indentify_cor_type.item()
    idx_ll = []
    for i, labels_names in enumerate(labels_names_ll):
        cor_artery_type = indentify_cor_type[i].item()
        if cor_artery_type == 0:
            idx = find_matches_two_lists(labels_names, identify_cor_artery_lca)
        elif cor_artery_type == 1:
            idx = find_matches_two_lists(labels_names, identify_cor_artery_rca)
        else:
            raise ValueError("Not supported with labels_predictions different from 0 and 1")
        idx_ll.append(idx)
    return idx_ll

def remove_outputs_from_loss(outputs, targets, cor_artery_type, labels_names):
    identify_cor_artery_rca = ['_1_prox',  '_2_mi' , '_3_dist', '_4_pda', '_16_pla']
    identify_cor_artery_lca = [ '_4_pda', '_5_lm', '_6_prox', '_7_mi', '_8_dist',
                            '_9_d1', '_10_d2', '_11_prox', '_12_om', '_13_midt', '_14_om', '_15_dist', '_16_pla']
    # test if cor_artery_type is contains a match in identify_cor_artery_rca or identify_cor_artery_lca
    
    if cor_artery_type.item() == 0:  # lca
        idx = find_matches_two_lists(labels_names, identify_cor_artery_lca)
        # remove the outputs from the loss
        outputs = outputs[:,idx]
        targets = targets[:,idx]
    elif cor_artery_type.item() == 1: # rca
        idx = find_matches_two_lists(labels_names, identify_cor_artery_rca)
        outputs = outputs[:, idx]
        targets = targets[:, idx]
    else:
        raise ValueError("Not supported with labels_predictions different from 0 and 1")
    return outputs, targets
def weighted_focal_l1_loss_multioutput(inputs, targets_tuple, weights=None, activate='sigmoid', beta=.2, gamma=1, train=True):
    loss = 0
   # torch.tensor(0.0, device=predictions.device)
    if not train:
        weights = None
    targets = targets_tuple[0]
    cor_artery_type = targets_tuple[1]
    labels_names = targets_tuple[2]
    config = targets_tuple[3]
    #"############################"
    # if config["task_type"] != "mil_classification":
    #     float_nan_tensor = create_nan_tensor_to_remove_segments_from_targets(labels_names, cor_artery_type.size(0), cor_artery_type)
    #     device = targets.get_device()
    #     device_type = targets.device.type
    #     # move to cuda
    #     float_nan_tensor = float_nan_tensor.to(device_type + ":" + str(device))
    #     targets = targets * float_nan_tensor
 
 ###################
       # inputs, targets = remove_outputs_from_loss(inputs, targets, cor_artery_type, labels_names)
    for i in range(0, inputs.shape[1]):
        if weights is not None:
            loss += weighted_focal_l1_loss(inputs[:,i],targets[:,i], weights=weights[:,i])
        else:
            loss += focal_l1_loss(inputs[:,i],targets[:,i])

    return loss
        
def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    mask = torch.isnan(targets)
    targets = targets[~mask]
    inputs = inputs[~mask]
    if weights is not None:
        weights = weights[~mask]

    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    if targets.size()==torch.Size([0]):
        print('should not happen')
        loss=torch.tensor(0.0, device=inputs.device)
    return loss


def focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    mask = torch.isnan(targets)
    targets = targets[~mask]
    inputs = inputs[~mask]
  
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    loss = torch.mean(loss)
    if targets.size()==torch.Size([0]):
        print('should not happen')
        loss=torch.tensor(0.0, device=inputs.device)
    return loss


def get_loss_func(config, train=True):
    criterions = []
    for loss in config['loss']['groups_names']:
        if loss.startswith('CE'):
            criterion = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=99998)
            criterions.append(criterion)
        elif loss.startswith('BCE_multilabel'):
            criterion = bce_with_nans
            criterion.__defaults__ = (config,train)

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
            weighted_focal_l1_loss_multioutput.__defaults__ = (
                weighted_focal_l1_loss_multioutput.__defaults__[0],
                weighted_focal_l1_loss_multioutput.__defaults__[1],
                weighted_focal_l1_loss_multioutput.__defaults__[2],
                weighted_focal_l1_loss_multioutput.__defaults__[3],
                train)
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
        elif loss.startswith('NNL_cont'):
            criterion = cox_ph_loss
            criterions.append(criterion)
        elif loss.startswith('NNL'):
          #  criterion = NLLSurvLoss(alpha=0.5, eps=1e-7, reduction='mean')
            criterion = nll_logistic_hazard
        #    criterion = gradient_blending_nnl_loss
            criterions.append(criterion)
        else:
            raise ValueError("Loss type is not implemented")
    return criterions
