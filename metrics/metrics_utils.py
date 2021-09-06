import os
from metrics.metrics import MeanIoU, softmax_transform, corrects_top, corrects_top_batch
import collections
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unroll_list_in_dict(config):
    for i in list(config):
        if isinstance(config[i], list):
            c = 0
            for intrance in config[i]:
                config[i+str(c)] = intrance
                c += 1
            del config[i]
    return config


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def init_metrics(metrics, config=None, mode="metric"):
    if 'dice_class' in metrics:
        for class_i in range(0, config['model']['num_classes']):
            metrics.append('dice_class_'+str(class_i))
        metrics.remove('dice_class')
    if mode == "loss":
        if len(metrics) > 1:
            metrics = ['total_loss'] + metrics
    dicts = {}
    keys = [0.0] * len(metrics)
    idx = range(len(keys))
    for i in idx:
        dicts[metrics[i]] = keys[i]

    return dicts, metrics


def write_tensorboard(losses, metrics, writer, tb_step_writer, phase):
    for loss in losses:
        writer.add_scalar("{}/{}".format(loss, phase),
                          losses[loss],
                          tb_step_writer)
    for metric in metrics:
        writer.add_scalar("{}/{}".format(metric, phase),
                          metrics[metric],
                          tb_step_writer)
    return None


def get_metrics(outputs, labels, metrics):
    dicts = {}
    for metric in metrics:
        c = 0
        if metric == 'acc_top_1':
            dicts[metric] = \
                corrects_top_batch(outputs, labels, topk=(1, ))[0].item()
        elif metric == 'acc_top_5':
            dicts[metric] = \
                corrects_top_batch(outputs, labels, topk=(1, 5))[1].item()
        elif metric == 'MeanIoU':  # does not work properly i think
            criterion = MeanIoU()
            dicts[metric] = criterion(softmax_transform(outputs), labels)
        elif metric == 'dice_global':
            post_trans_multiCat = Compose(
                [Activations(softmax=True),
                 AsDiscrete(
                     argmax=True, to_onehot=True,
                     n_classes=labels.shape[1]),
                     ])
            outputs = post_trans_multiCat(outputs)
            dice_global = DiceMetric(include_background=True,
                                     reduction="mean")
            dicts[metric] = dice_global(outputs, labels)[0]

        elif metric.startswith('dice_class_'):
            if c < 1:
                post_trans_multiCat = Compose(
                    [Activations(softmax=True),
                     AsDiscrete(
                        argmax=True, to_onehot=True,
                        n_classes=labels.shape[1])])
                outputs = post_trans_multiCat(outputs)
                dice_channel = DiceMetric(include_background=True,
                                          reduction="mean_batch")
                dice_channel_result = dice_channel(outputs, labels)[0]
                for class_id in range(0, labels.shape[1]):
                    dicts[metric[:-1]+str(class_id)] = \
                        dice_channel_result[class_id]
                c += 1
        else:
            raise ValueError("Invalid metric %s" % repr(metric))
    return dicts


def normalize_metrics(running_metrics, config, data_len):
    for running_metric in running_metrics:
        running_metrics[running_metric] = running_metrics[running_metric] \
            / (data_len)
    return running_metrics


def increment_metrics(running_metrics, metrics, config=None):
    for metric in metrics:
        running_metrics[metric] += metrics[metric]
    return running_metrics


def create_loss_dict(config, losses):
    if len(losses) == 1:
        losses = dict(zip(config['loss']['name'], losses))
    else:
        losses = dict(zip(['total_loss'] + config['loss']['name'], losses))
    return losses
