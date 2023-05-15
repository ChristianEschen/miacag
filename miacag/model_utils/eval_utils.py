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
        outputs = model(inputs)
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
    # set model in eval mode
    model.eval()
    with torch.no_grad():
        # forward
        outputs = maybe_sliding_window(data['inputs'], model, config)
        losses, loss = get_losses_class(config, outputs,
                                        data, criterion, device)
        outputs = wrap_outputs_to_dict(outputs, config)

        losses = create_loss_dict(config, losses, loss)
        metrics, losses_metric = get_loss_metric_class(config, outputs,
                                                        data, losses,
                                                        running_metric_val,
                                                        running_loss_val,
                                                        criterion)
 
    # if config['loaders']['val_method']['saliency'] == 'True':
    #     cams = calc_saliency_maps(model, data['inputs'], config, )
    #     return outputs, losses, metrics, cams
    # else:
    return outputs, losses, metrics, None

def forward_model(inputs, model, config):
    if config['loaders']['use_amp'] is True:
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
    else:
        outputs = model(inputs)
    return outputs


def eval_one_step_knn(get_data_from_loader,
                      validation_loader,
                      model, device, criterion,
                      config, saliency_maps=False):
    train_loader = validation_loader[1]
    val_loader = validation_loader[2]
    batch_size = config['loaders']['batchSize']
    n_data = len(train_loader)*batch_size
    K = 1
    if config['cpu'] == "False":
        encoder_model = model.module.encoder_projector
    else:
        encoder_model = model.encoder_projector
    # set model in eval mode
    encoder_model.eval()
    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    train_features = torch.zeros([config['model']['feat_dim'], n_data],
                                 device=device)
    train_data['labels'] = torch.zeros([config['model']['feat_dim'], n_data],
                                 device=device)
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            inputs, data['labels'] = get_data_from_loader(data, config,
                                                  device, val_phase=True)
            # forward
            features = forward_model(inputs, encoder_model, config)
            features = nn.functional.normalize(features)
            train_features[:,
                           batch_idx * batch_size:batch_idx
                           * batch_size + batch_size] = features.data.t()
            train_labels[:,
                         batch_idx * batch_size:batch_idx
                         * batch_size + batch_size] = labels.data.t()

    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            inputs, data['labels'] = get_data_from_loader(data, config,
                                                  device, val_phase=True)
            features = forward_model(inputs, encoder_model, config)
            features = features.type(torch.cuda.FloatTensor)
            dist = torch.mm(features, train_features)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

            total += labels.size(0)
            correct += retrieval.eq(labels.data).sum().item()
    top1 = correct / total
    return top1


def set_uniform_sample_pct(validation_loader, percentage):
    for i in validation_loader.dataset.transform.transforms:
        if hasattr(i, 'percentage'):
            i.percentage = percentage
    return validation_loader


def run_val_one_step(model, config, validation_loader, device, criterion,
                     saliency_maps, running_metric_val,
                     running_loss_val):
    if config['task_type'] != "representation_learning":
        # initialize pandas dataframe from dict
        df_results = pd.DataFrame()
        
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
                outputs = {key: value.cpu().numpy().tolist()
                           for (key, value) in outputs.items()}
                outputs['rowid'] = data['rowid'].cpu().numpy().tolist()
                nr_values = len(outputs[[i for i in outputs][0]])
                for i in range(0, nr_values):
                    outputs_i = {
                        key: value[i] for (key, value) in outputs.items()}
                    # append outputs to dataframe
                    df_dictionary = pd.DataFrame([outputs_i])
                    df_results = pd.concat(
                        [df_results, df_dictionary], ignore_index=True)

    if config['loaders']['mode'] == 'training':
        return running_metric_val, running_loss_val, None, None
    else:
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
            running_metric_val)

    running_loss_val, loss_tb = normalize_metrics(
        running_loss_val)

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
    df_result = pd.DataFrame()
    samples = config['loaders']['val_method']["samples"]
    for i in range(0, samples):
        eval_outputs = run_val_one_step(
                model, config, validation_loader, device, criterion,
                saliency_maps,
                running_metric_val, running_loss_val)
        running_metric_val, running_loss_val, df_result_i, _ = eval_outputs

        df_result = pd.concat(
                        [df_result, df_result_i], ignore_index=True)

    if config['task_type'] != "representation_learning":
        running_metric_val, metric_tb = normalize_metrics(
            running_metric_val)

    running_loss_val, loss_tb = normalize_metrics(
        running_loss_val)
    df_result = maybe_softmax_transform(df_result, config)
    return metric_tb, df_result


def maybe_softmax_transform(df, config):
    cols = df.columns
    # remove rowid from the list
    cols = cols.drop('rowid')
    logits_return = []
    for c, logit in enumerate(cols):
        logit_conf = logit + '_confidence'
        if config['loss']['name'][c].startswith('CE'):
            raise(ValueError('this transform is not implemented'))
            logits_return.append(softmax_transform(logit.float()))
        elif config['loss']['name'][c] == 'MSE':
            logits_return.append(logit.float())
        elif config['loss']['name'][c] in ['_L1', 'L1smooth']:
            df[logit_conf] = df[logit]
        elif config['loss']['name'][c].startswith('BCE'):
            logits_return.append(torch.nn.Sigmoid()(logit.float()))
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
        metric_tb, df_results = val_one_epoch_test(
            model, criterion, config,
            validation_loader, device,
            running_metric_val, running_loss_val,
            writer, epoch, saliency_maps)
        return metric_tb, df_results  # predictions
