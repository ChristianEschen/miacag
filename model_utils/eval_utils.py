from metrics.metrics_utils import normalize_metrics, get_metrics, \
    create_loss_dict, increment_metrics, write_tensorboard
import torch
from dataloader.get_dataloader import get_data_from_loader
from monai.inferers import sliding_window_inference

from monai.inferers import SlidingWindowInferer
from torch import nn


def  eval_one_step(model, inputs, labels, device, criterion,
                  config, saliency_maps=False):
    if saliency_maps:
        model = build_model(config, device)
    # set model in eval mode
    model.eval()
    with torch.no_grad():
        # forward
        if config['loaders']['val_method']['type'] == 'sliding_window':
            if config['model']['dimensions'] == 3:
                input_shape = (config['loaders']['height'],
                               config['loaders']['width'],
                               config['loaders']['depth'])
            elif config['model']['dimensions'] == 2:
                input_shape = (config['loaders']['height'],
                               config['loaders']['width'])
            if config['loaders']['use_amp'] is True:
                with torch.cuda.amp.autocast():
                    outputs = sliding_window_inference(
                            inputs, input_shape,
                            1, model)
            else:
                outputs = sliding_window_inference(
                            inputs, input_shape,
                            1, model)
        elif config['loaders']['val_method']['type'] in [
                'patches', 'image_lvl',
                'image_lvl+saliency_maps', 'saliency_maps']:
            if config['loaders']['use_amp'] is True:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
        else:
            raise ValueError("Invalid validation moode %s" % repr(
                config['loaders']['val_method']['type']))

        losses, _ = get_losses(config, outputs, labels, criterion)
        losses = create_loss_dict(config, losses)
    if config['loaders']['task_type'] == "representation_learning":
        return outputs, losses, _
    else:
        metrics = get_metrics(outputs, labels,
                              config['eval_metric_val']['name'])
    return outputs, losses, metrics


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
    if config['cpu'] is False:
        model = model.module.encoder,
    else:
        model = model.encoder
    # set model in eval mode
    model.eval()
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
            features = forward_model(inputs, model, config)
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
            features = forward_model(inputs, model, config)
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
        for data in validation_loader:
            inputs, labels = get_data_from_loader(data, config,
                                                    device)
            _, loss, metrics = eval_one_step(
                                            model, inputs,
                                            labels, device,
                                            criterion,
                                            config, saliency_maps)
            running_metric_val = increment_metrics(running_metric_val,
                                                    metrics)
            running_loss_val = increment_metrics(loss, running_loss_val)
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
            running_loss_val = increment_metrics(loss, running_loss_val)

    return running_metric_val, running_loss_val


def val_one_epoch(model, criterion, config,
                  validation_loader, device,
                  running_metric_val=0.0, running_loss_val=0.0,
                  writer=False, epoch=0, saliency_maps=False):
    if config['loaders']['format'] == 'avi':
        samples = config['loaders']['val_method']['samples']
        frames_sample_list = [
            i*0.1 for i in range(0, samples)]
        for sample in range(0, samples):
            validation_loader = set_uniform_sample_pct(
                validation_loader, frames_sample_list[sample])
    else:
        running_metric_val, running_loss_val = run_val_one_step(
                model, config, validation_loader, device, criterion,
                saliency_maps,
                running_metric_val, running_loss_val)
        samples = 1

    # Normalize the metrics from the entire epoch
    if config['loaders']['task_type'] != "representation_learning":
        running_metric_val = normalize_metrics(
            running_metric_val,
            len(validation_loader)*samples)

    running_loss_val = normalize_metrics(
        running_loss_val,
        len(validation_loader)*samples)
    if writer is not False:
        write_tensorboard(running_loss_val,
                          running_metric_val,
                          writer, epoch, 'val')
    else:
        print('metrics', running_metric_val)
        print('loss', running_loss_val)
    running_metric_val.update(running_loss_val)
    return running_metric_val
