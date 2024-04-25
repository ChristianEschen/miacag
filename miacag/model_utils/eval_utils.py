from miacag.metrics.metrics_utils import normalize_metrics, get_metrics, \
    create_loss_dict, write_tensorboard, get_losses_metric, \
    mkDir, get_loss_metric_class
import torch
from miacag.dataloader.get_dataloader import get_data_from_loader
from monai.inferers import sliding_window_inference

from monai.inferers import SlidingWindowInferer
from torch import nn
from miacag.metrics.metrics import softmax_transform
from miacag.model_utils.grad_cam_utils import prepare_cv2_img
import numpy as np
from monai.visualize import CAM, GradCAM
from miacag.model_utils.grad_cam_utils import calc_saliency_maps
from miacag.models.modules import getCountsLoss, unique_counts
from miacag.utils.common_utils import get_losses_class, wrap_outputs_to_dict, get_loss
import pandas as pd
import os
import psutil




def check_memory():
    pid = os.getpid()
    proc = psutil.Process(pid)

    # Get process memory info in bytes
    mem_info = proc.memory_info()
    mem_in_gb = mem_info.rss / (1024 ** 3)  # Convert to GB

    threshold_in_gb = 100  # Define your threshold here

    if mem_in_gb > threshold_in_gb:
        raise MemoryError('Memory usage of this process is over {} GB'.format(threshold_in_gb))


def get_input_shape(config):
    if config['model']['dimension'] in ['2D+T', 3]:
        input_shape = (config['loaders']['Crop_height'],
                       config['loaders']['Crop_width'],
                       config['loaders']['Crop_depth'])
    elif config['model']['dimension'] == 2:
        input_shape = (config['loaders']['Crop_height'],
                       config['loaders']['Crop_width'])
    else:
        raise ValueError("Invalid dimension %s" % repr(
            config['model']['dimension']))
    return input_shape


def maybe_sliding_window(inputs, model, config):
    if config['loaders']['val_method']['type'] == 'sliding_window' \
            and config['task_type'] == "segmentation":
        input_shape = get_input_shape(config)
        outputs = sliding_window_inference(inputs, input_shape, 1, model)
    else:
        outputs = maybe_use_amp(config['loaders']['use_amp'], inputs, model)
    return outputs


def maybe_use_amp(use_amp, inputs, model):
    if use_amp is True:
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
    else:
        outputs = model(inputs)
    return outputs


def eval_one_step(model, data, device, criterion,
                  config, running_metric_val, running_loss_val,
                  saliency_maps=False):
    import time
    model.eval()
    with torch.no_grad():
        # forward
        
        outputs = maybe_sliding_window(data['inputs'], model, config)
        # max pooling on first dimension
        if config['loaders']['mode'] == 'testing':
            outputs = [torch.unsqueeze(torch.mean(outputs[0], dim=0), dim=0)]
            # if config['labels_names'][0].startswith('ffr'):
            #     outputs = [torch.unsqueeze(torch.amin(outputs[0], dim=0), dim=0)]
            # elif config['labels_names'][0].startswith('sten'):

            #     outputs = [torch.unsqueeze(torch.amax(outputs[0], dim=0), dim=0)]
            # else:
            #     raise(ValueError('this is not implemented'))
        if config['loaders']['mode'] != 'testing':
            losses, loss = get_losses_class(config, outputs,
                                            data, criterion, device)
            losses = create_loss_dict(config, losses, loss)
        outputs = wrap_outputs_to_dict(outputs, config)
        if config['loaders']['mode'] != 'testing':
            metrics, losses_metric = get_loss_metric_class(config, outputs,
                                                            data, losses,
                                                            running_metric_val,
                                                            running_loss_val,
                                                            criterion)
            return outputs, losses, metrics, None
        else:
            return outputs, None, None, None

def forward_model(inputs, model, config):
    if config['loaders']['use_amp'] is True:
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
    else:
        outputs = model(inputs)
    return outputs


def set_uniform_sample_pct(validation_loader, percentage):
    for i in validation_loader.dataset.transform.transforms:
        if hasattr(i, 'percentage'):
            i.percentage = percentage
    return validation_loader


def run_val_one_step(model, config, validation_loader, device, criterion,
                     saliency_maps, running_metric_val,
                     running_loss_val):

    df_results_list = []
    #config['loaders']['batchSize']= 2
    model.eval()
    with torch.no_grad():
        for data in validation_loader:
            data = get_data_from_loader(data, config, device)
            outputs, loss, metrics, cams = eval_one_step(
                                            model, data, device,
                                            criterion,
                                            config,
                                            running_metric_val,
                                            running_loss_val,
                                            saliency_maps)

            if config['loaders']['mode'] == 'testing':
                # list comprehension on the dict outputs
                outputs_i_i = []
                for o_idx in range(0, len(data['rowid'])):
                    outputs_i = {key: value.cpu().numpy().tolist()[o_idx]
                                for (key, value) in outputs.items()}
                    outputs_i['rowid'] = data['rowid'].cpu().numpy().tolist()[o_idx]
                    outputs_i_i.append(outputs_i)
                df_results_list.append(outputs_i_i)

    if config['loaders']['mode'] == 'training':
        return running_metric_val, running_loss_val, None, None
    else:
         # Convert the list of dictionaries into a pandas DataFrame
        flattened_list = [dict_item for sublist in df_results_list for dict_item in sublist]

        # Convert the list of dictionaries into a pandas DataFrame
        df_results = pd.DataFrame(flattened_list)
        return running_metric_val, running_loss_val, df_results, None


def val_one_epoch_train(
        model, criterion, config,
        validation_loader, device,
        running_metric_val=0.0, running_loss_val=0.0,
        writer=False, epoch=0, saliency_maps=False):
    
    eval_outputs = run_val_one_step(
            model, config, validation_loader, device, criterion,
            saliency_maps,
            running_metric_val, running_loss_val)
    running_metric_val, running_loss_val, _, _ = eval_outputs

    # Normalize the metrics from the entire epoch
    if config['task_type'] != "representation_learning":
        running_metric_val, metric_tb = normalize_metrics(
            running_metric_val, device)

    running_loss_val, loss_tb = normalize_metrics(
        running_loss_val, device)

    if writer is not False:
        loss_tb, metric_tb = write_tensorboard(
            loss_tb,
            metric_tb,
            writer, epoch, 'val')

    metric_tb.update(loss_tb)
    return metric_tb


def val_one_epoch_test(
        model, criterion, config,
        validation_loader, device,
        running_metric_val=0.0, running_loss_val=0.0,
        writer=False, epoch=0, saliency_maps=False):
   # df_result = pd.DataFrame()
    df_results_list = []
    samples = config['loaders']['val_method']["samples"]
    if samples > 1:
        raise(ValueError('this is not implemented'))
    for i in range(0, samples):
        eval_outputs = run_val_one_step(
                model, config, validation_loader, device, criterion,
                saliency_maps,
                running_metric_val, running_loss_val)
        running_metric_val, running_loss_val, df_result_i, _ = eval_outputs
        df_results_list.append(df_result_i)
        # df_result = pd.concat(
        #                 [df_result, df_result_i], ignore_index=True)
    df_result = pd.concat(df_results_list)
    if config['loaders']['mode'] != 'testing':
        running_metric_val, metric_tb = normalize_metrics(
            running_metric_val, device)
    df_result = maybe_softmax_transform(df_result, config)
    if config['loaders']['mode'] != 'testing':
        return metric_tb, df_result
    else:
        return None, df_result


def maybe_softmax_transform(df, config):
    cols = df.columns
    # remove rowid from the list
    cols = cols.drop('rowid')
    logits_return = []
    for c, logit in enumerate(cols):
        logit_conf = logit + '_confidence'
        if config['loss']['name'][c].startswith('CE'):
            df[logit_conf] = df[logit]
            #raise(ValueError('this transform is not implemented'))
            df[logit_conf] = softmax_transform(torch.tensor(df[logit])).tolist()
        elif config['loss']['name'][c] == 'MSE':
            logits_return.append(logit.float())
        elif config['loss']['name'][c] in ['_L1', 'L1smooth', 'wfocall1']:
            df[logit_conf] = df[logit]
        elif config['loss']['name'][c].startswith('BCE'):
            logits_return.append(torch.nn.Sigmoid()(logit.float()))
        elif config['loss']['name'][c].startswith('NNL'):
            df[logit_conf] = df[logit]
        else:
            raise(ValueError('this loss type is not implemented'))
    return df


def getListOfLogits(logits):
    unrolled_logits = []
    for logit in logits:
        for output_idx in logit[0]:
            unrolled_logits.append(output_idx)
    unrolled_logits = torch.vstack(unrolled_logits)
    return [unrolled_logits]
    # label_liste = []
    # for lo in logits:
    #     for label in lo:
    #         label_liste.append(label)
    # label_liste = np.array(label_liste)
    # uniques = list(range(0, len(label_names)))
    # idxes = uniques*data_len
    # idxes = np.array(idxes)

    # list_logits = []
    # for un in uniques:
    #     un_np_idx = np.where(idxes == un)
    #     list_logits.append(torch.vstack(list(label_liste[un_np_idx])))
    # return list_logits


def val_one_epoch(model, criterion, config,
                  validation_loader, device,
                  running_metric_val=0.0, running_loss_val=0.0,
                  writer=False, epoch=0, saliency_maps=False):
    if config['loaders']['mode'] == 'training':
        metric_tb = val_one_epoch_train(
            model, criterion, config,
            validation_loader, device,
            running_metric_val, running_loss_val,
            writer, epoch, saliency_maps)
        return metric_tb

    else:
        try:
            check_memory()
            metric_tb, df_results = val_one_epoch_test(
                model, criterion, config,
                validation_loader, device,
                running_metric_val, running_loss_val,
                writer, epoch, saliency_maps)
        except MemoryError as e:
            print(e)
        
        # maybe return survival predictions
       # if config['loss']['name'][0] == 'NNL':
            # logits = np.concatenate(
            #     [np.expand_dims(np.array(i),-1) for i in df_results[config['labels_names'][0] + '_confidence']],
            #     axis=1)
            # surv = predict_surv(logits.T, config['cuts'])
            # surv = surv.T.values.tolist()
            # df_results[config['labels_names'][0] + '_confidence']= surv
        return metric_tb, df_results  # predictions
