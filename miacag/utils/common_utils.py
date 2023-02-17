import torch

def stack_labels(data, config, loss_name):
    stacked_data = []
    for count_idx, label_name in enumerate(config['labels_names']):
        if label_name.partition("_")[0] == loss_name.partition("_")[-1]:
            if loss_name.startswith(tuple(['MSE', '_L1', 'L1smooth'])):
                stacked_data.append(data[label_name])
            elif loss_name.startswith(tuple(['BCE_multilabel'])):
                stacked_data.append(data[label_name])
            elif loss_name.startswith(tuple(['CE'])):
                stacked_data.append(data[label_name])
            else:
                raise ValueError('this loss is not implementeed:', loss_name)
    return torch.stack(stacked_data, 1)


#
def wrap_outputs_to_dict(outputs, config):
    outputs_dict = {}
    for group_count, group in enumerate(config['loss']['groups_names']):
        outputs_group = outputs[group_count]
        for segment_idx in range(0, outputs_group.shape[-1]):
            output_segment = outputs_group[:, segment_idx]
            label_name_idx = config['loss'][
                'group_idx']['loss_group'][group_count][segment_idx]
            output_name = config['labels_names'][label_name_idx]
            outputs_dict[output_name] = output_segment
    return outputs_dict


# this function is used to get the loss for each task
def get_losses_class(config, outputs, data, criterion, device):
    losses = []
    loss_tot = torch.tensor([0]).float()
    loss_tot = loss_tot.to(device)
    loss_tot = loss_tot.requires_grad_()
    for count_idx, loss_name in enumerate(config['loss']['groups_names']):
        labels = stack_labels(data, config, loss_name)
        loss = get_loss(
            config, outputs[count_idx],
            labels, criterion[count_idx], loss_name)
        if torch.isnan(loss) == torch.tensor(True, device=device):
            raise ValueError('the loss is nan!')

        else:
            # scale loss by weights for given task
            loss = loss * config['groups_weights'][count_idx]
            losses.append(loss)
            loss_tot = loss_tot + loss
    losses = [loss_indi.item() for loss_indi in losses]
    losses = losses + [loss_tot.item()]
    return losses, loss_tot

# get loss function
def get_loss(config, outputs, labels, criterion, loss_name):
    if 'Siam' in config['loss']['name']:
        loss = criterion(outputs)
    elif loss_name.startswith('CE'):
        labels = torch.reshape(labels, (labels.shape[0], ))
        loss = criterion(outputs, labels)
    else:
        loss = criterion(outputs, labels)
    return loss
