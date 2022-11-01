import torch


def stack_labels(data, config, loss_name):
    stacked_data = []
    for count_idx, label_name in enumerate(config['labels_names']):
        loss_for_label = config['loss']['name'][count_idx]
        if loss_name == loss_for_label:
            if loss_for_label in ['MSE', 'L1', 'L1smooth']:
                stacked_data.append(data[label_name])
            elif loss_for_label in ['BCE_multilabel']:
                stacked_data.append(data[label_name])
            elif loss_for_label.startswith('CE'):
                stacked_data.append(data[label_name])
             #   print('not impleneted jet!!')
             
            else:
                raise ValueError('this loss is not implementeed:', loss_for_label)
    return torch.stack(stacked_data, 1)