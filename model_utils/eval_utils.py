from metrics.metrics_utils import normalize_metrics, get_metrics, \
    create_loss_dict, increment_metrics, write_tensorboard
import torch
from dataloader.get_dataloader import get_data_from_loader
from monai.inferers import sliding_window_inference

from monai.inferers import SlidingWindowInferer
from models.image2scalar_utils.utils_3D.test_utils import build_model


def eval_one_step(model, inputs, labels, device, criterion,
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
        losses = [crit(outputs, labels) for crit in criterion]
        loss = torch.stack(losses, dim=0).sum(dim=0)
        if len(losses) > 1:
            losses = [loss] + losses
        losses = [l.item() for l in losses]
        # add loss dict
        metrics = get_metrics(outputs, labels, config['eval_metric']['name'])
        losses = create_loss_dict(config, losses)
    return outputs, losses, metrics


def set_uniform_sample_pct(validation_loader, percentage):
    for i in validation_loader.dataset.transform.transforms:
        if hasattr(i, 'percentage'):
            i.percentage = percentage
    return validation_loader


def run_val_one_step(model, config, validation_loader, device, criterion,
                     saliency_maps, running_metric_val,
                     running_loss_val):
    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = get_data_from_loader(data, config, device)
            _, loss, metrics = eval_one_step(model, inputs,
                                             labels, device,
                                             criterion,
                                             config, saliency_maps)

            running_metric_val = increment_metrics(running_metric_val, metrics)
            running_loss_val = increment_metrics(loss, running_loss_val)
    return running_metric_val, running_loss_val


def val_one_epoch(model, criterion, config,
                  validation_loader, device,
                  running_metric_val=0.0, running_loss_val=0.0,
                  writer=False, epoch=0, saliency_maps=False):
    if config['model_name'] in ['ir_csn_152_', 'ip_csn_152_']:
        if config['loaders']['format'] == 'avi':
            samples = config['loaders']['val_method']['samples']
            frames_sample_list = [
                i*0.1 for i in range(0, samples)]
            for sample in range(0, samples):
                validation_loader = set_uniform_sample_pct(
                    validation_loader, frames_sample_list[sample])

                running_metric_val, running_loss_val = run_val_one_step(
                    model, config, validation_loader, device, criterion,
                    saliency_maps,
                    running_metric_val, running_loss_val)
        elif config['loaders']['format'] == 'nifty':
            running_metric_val, running_loss_val = run_val_one_step(
                model, config, validation_loader, device, criterion,
                saliency_maps, running_metric_val, running_loss_val)
            samples = 1

    elif config['model_name'] in ['UNet2D', 'UNet3D', 'DYNUNet3D']:
        running_metric_val, running_loss_val = run_val_one_step(
            model, config, validation_loader, device, criterion,
            saliency_maps, running_metric_val, running_loss_val)
        samples = 1
    else:
        raise ValueError("Invalid model name %s" % repr(
            config['model_name']))

    # Normalize the metrics from the entire epoch
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
