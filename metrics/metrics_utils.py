import os
from metrics.metrics import MeanIoU, softmax_transform, corrects_top, corrects_top_batch
import collections
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.metrics import ConfusionMatrixMetric
from monai.data import decollate_batch
import torch.nn.functional as F
import monai
import torch
from metrics.cumulativeSumming import CumulativeSumming
from monai.metrics import Cumulative, CumulativeAverage, CumulativeIterationMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToDeviced,
    EnsureTyped,
    EnsureType,
)

def convert_dict_to_str(labels_dict_val):
    items = []
    for k, v in labels_dict_val.items():
        k = str(k)
        v = str(v)
        items.append((k, v))
    return dict(items)


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        if k == 'labels_dict':
            v = convert_dict_to_str(v)
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


def init_metrics(metrics):
    dicts = {}
    keys = [0.0] * len(metrics)
    idx = range(len(keys))
    for i in idx:
        if metrics[i] == 'CE':
            dicts[metrics[i]] = CumulativeAverage()
            #dicts[metrics[i]] = CumulativeSumming()
        elif metrics[i] == 'acc_top_1':
            #dicts[metrics[i]] = CumulativeSumming()
            dicts[metrics[i]] = ConfusionMatrixMetric(
                metric_name='accuracy', reduction="mean", include_background=False)
        else:
            raise NotImplementedError(
                'This metric {} is not implemented!'.format(metrics[i]))
    return dicts


def write_tensorboard(losses, metrics, writer, tb_step_writer, phase):
    if torch.distributed.get_rank() == 0:
        for loss in losses:
            writer.add_scalar("{}/{}".format(loss, phase),
                              losses[loss],  # losses[loss],
                              tb_step_writer)
        for metric in metrics:
            writer.add_scalar("{}/{}".format(metric, phase),
                              metrics[metric],
                              tb_step_writer)
    return losses, metrics

def get_metrics(outputs,
                labels,
                metrics,
                criterion,
                config):
    dicts = {}
    for metric in metrics:
        c = 0
        if metric == 'acc_top_1':
            post_trans = Compose(
                [EnsureType(),
                 Activations(softmax=True),
                 AsDiscrete(threshold=0.5)]
                 )
            outputs = [post_trans(i) for i in decollate_batch(outputs)]
            labels = F.one_hot(
                labels,
                num_classes=config['model']['num_classes'])
            metrics[metric](y_pred=outputs, y=labels)
             
            #corrects = torch.sum(torch.stack(outputs)*labels)
           # metrics[metric].append(corrects.item())
            dicts[metric] = metrics[metric]
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


def get_losses_metric(outputs,
                      labels,
                      running_losses,
                      losses,
                      criterion,
                      config):
    dicts = {}
    for loss in losses:
        if loss == 'CE':
            running_losses[loss].append(losses[loss])
            dicts[loss] = running_losses[loss]
        else:
            raise ValueError("Invalid loss %s" % repr(loss))
    return dicts
# OLD
# def normalize_metrics(running_metrics, config, data_len):
#     for running_metric in running_metrics:
#         value = torch.sum(
#             running_metrics[running_metric].get_buffer()) / data_len
#         value = value.item()
#         running_metrics[running_metric] = value
#     return running_metrics

# NEW
def normalize_metrics(running_metrics):
    metric_dict = {}
    for running_metric in running_metrics:
        if running_metric == 'CE':
            metric_tb = running_metrics[running_metric].aggregate().item()
        else:
           # metric_tb = running_metrics[running_metric].aggregate()[0].item()
           # metric_tb = running_metrics[running_metric].sum.item()
            metric_tb = running_metrics[running_metric].aggregate()[0].item()
        metric_dict[running_metric] = metric_tb
        running_metrics[running_metric].reset()
    return running_metrics, metric_dict


def create_loss_dict(config, losses):
    if len(losses) == 1:
        losses = dict(zip(config['loss']['name'], losses))
    else:
        losses = dict(zip(['total_loss'] + config['loss']['name'], losses))
    return losses
