from metrics.metrics_utils import normalize_metrics, get_metrics, \
    create_loss_dict, write_tensorboard, get_losses_metric, \
    mkDir
import torch
from dataloader.get_dataloader import get_data_from_loader
from monai.inferers import sliding_window_inference

from monai.inferers import SlidingWindowInferer
from monai.inferers import SimpleInferer, SaliencyInferer
from torch import nn
from metrics.metrics import softmax_transform
from model_utils.grad_cam_utils import prepare_cv2_img


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


def eval_one_step(model, inputs, labels, device, criterion,
                  config, running_metric_val, running_loss_val,
                  saliency_maps=False):
    # set model in eval mode
    model.eval()
    with torch.no_grad():
        # forward
        outputs = maybe_sliding_window(inputs, model, config)

        losses, _ = get_losses(config, outputs, labels, criterion)
        losses = create_loss_dict(config, losses)
    if config['task_type'] == "representation_learning":
        return outputs, losses, _
    else:
        metrics = get_metrics(outputs,
                            labels,
                            running_metric_val,
                            criterion,
                            config)
        losses_metric = get_losses_metric(
            outputs,
            labels,
            running_loss_val,
            losses,
            criterion,
            config)
    
    if config['loaders']['val_method']['saliency'] == 'True':
        if config['loaders']['use_amp'] is True:
            with torch.cuda.amp.autocast():
                saliency = SaliencyInferer(
                    cam_name="GradCAM",
                    target_layers='module.encoder.6')
        else:
            saliency = SaliencyInferer(
                    cam_name="GradCAM",
                    target_layers='module.encoder.5.post_conv')
        cams = saliency(network=model, inputs=inputs)

        return outputs, losses, metrics, cams
    else:
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
    train_labels = torch.zeros([config['model']['feat_dim'], n_data],
                                 device=device)
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = get_data_from_loader(data, config,
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
            inputs, labels = get_data_from_loader(data, config,
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


def get_losses(config, outputs, labels, criterion):
    if 'Siam' in config['loss']['name']:
        losses = [crit(outputs) for crit in criterion]
    else:
        losses = [crit(outputs, labels) for crit in criterion]
    loss = torch.stack(losses, dim=0).sum(dim=0)
    if len(losses) > 1:
        losses = [loss] + losses
    losses = [l.item() for l in losses]
    return losses, loss


def set_uniform_sample_pct(validation_loader, percentage):
    for i in validation_loader.dataset.transform.transforms:
        if hasattr(i, 'percentage'):
            i.percentage = percentage
    return validation_loader


def run_val_one_step(model, config, validation_loader, device, criterion,
                     saliency_maps, running_metric_val,
                     running_loss_val):
    if config['task_type'] != "representation_learning":
        logits = []
        rowids = []
        for data in validation_loader:
            if config['loaders']['mode'] == 'testing':
                inputs, labels, rowid = get_data_from_loader(data, config,
                                                    device)
            else:
                inputs, labels = get_data_from_loader(data, config,
                                                    device)
            outputs, loss, metrics, cams = eval_one_step(
                                            model, inputs,
                                            labels, device,
                                            criterion,
                                            config,
                                            running_metric_val,
                                            running_loss_val,
                                            saliency_maps)
            if config['loaders']['mode'] == 'testing':
                logits.append(outputs.cpu())
                rowids.append(rowid.cpu())
            if config['loaders']['val_method']['saliency'] == 'True':
                patientID = data['DcmPathFlatten_meta_dict']['0010|0020'][0]
                studyInstanceUID = data['DcmPathFlatten_meta_dict']['0020|000d'][0]
                seriesInstanceUID = data['DcmPathFlatten_meta_dict']['0020|000e'][0]
                SOPInstanceUID = data['DcmPathFlatten_meta_dict']['0008|0018'][0]

                prepare_cv2_img(
                    inputs.cpu().numpy(),
                    cams.cpu().numpy(),
                    patientID,
                    studyInstanceUID,
                    seriesInstanceUID,
                    SOPInstanceUID,
                    config)



    else:
        metric = eval_one_step_knn(
            get_data_from_loader,
            validation_loader,
            model,
            device,
            criterion,
            config, saliency_maps)
        running_metric_val[config['eval_metric_val']['name'][0]] = metric

        for data in validation_loader[0]:
            inputs, labels = get_data_from_loader(data, config,
                                                    device)
            _, loss, _ = eval_one_step(
                                            model, inputs,
                                            labels, device,
                                            criterion,
                                            config, saliency_maps)
            # running_metric_val = increment_metrics(running_metric_val,
            #                                         metrics)
            #running_loss_val = increment_metrics(loss, running_loss_val)

    if config['loaders']['mode'] == 'training':
        return running_metric_val, running_loss_val, None, None
    else:
        return running_metric_val, running_loss_val, logits, rowids


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
    # running_metric_vals = []
    # running_loss_vals = []
    logitsS = []
    rowidsS = []
    for i in range(0, config['loaders']['val_method']["samples"]):
        eval_outputs = run_val_one_step(
                model, config, validation_loader, device, criterion,
                saliency_maps,
                running_metric_val, running_loss_val)
        running_metric_val, running_loss_val, logits, rowids = eval_outputs
        # running_metric_vals.append(running_metric_val)
        # running_loss_vals.append(running_loss_val)
        logitsS.append(logits)
        rowidsS.append(rowids)
    logitsS = [item for sublist in logitsS for item in sublist]
    rowidsS = [item for sublist in rowidsS for item in sublist]
    logits = torch.cat(logitsS, dim=0)
    rowids = torch.cat(rowidsS, dim=0)
    if config['task_type'] != "representation_learning":
        running_metric_val, metric_tb = normalize_metrics(
            running_metric_val)

    running_loss_val, loss_tb = normalize_metrics(
        running_loss_val)
    confidences = softmax_transform(logits.float())
    return metric_tb, confidences, rowids


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
        metric_tb, confidences, rowid = val_one_epoch_test(
            model, criterion, config,
            validation_loader, device,
            running_metric_val, running_loss_val,
            writer, epoch, saliency_maps)
        return metric_tb, confidences, rowid  # predictions

### GRAVEYARD


# def val_one_epoch(model, criterion, config,
#                   validation_loader, device,
#                   running_metric_val=0.0, running_loss_val=0.0,
#                   writer=False, epoch=0, saliency_maps=False):
#     #e  
#     for sample in range(0, config['loaders']['val_method']["samples"]):
#         eval_outputs = run_val_one_step(
#                 model, config, validation_loader, device, criterion,
#                 saliency_maps,
#                 running_metric_val, running_loss_val)
#     if config['loaders']['mode'] == 'training':
#         running_metric_val, running_loss_val, _, _ = eval_outputs
#     else:
#         running_metric_val, running_loss_val, logits, rowid = eval_outputs
#         confidences = softmax_transform(logits.float())

#     # Normalize the metrics from the entire epoch
#     if config['task_type'] != "representation_learning":
#         running_metric_val, metric_tb = normalize_metrics(
#             running_metric_val)

#     running_loss_val, loss_tb = normalize_metrics(
#         running_loss_val)

#     if writer is not False:
#         loss_tb, metric_tb = write_tensorboard(
#             loss_tb,
#             metric_tb,
#             writer, epoch, 'val')

#     metric_tb.update(loss_tb)


#     if config['loaders']['mode'] == 'training':
#         return metric_tb
#     else:
#         return metric_tb, confidences, rowid  # predictions