import torch
import numpy as np
import random
from metrics.metrics_utils import get_metrics, increment_metrics
from metrics.metrics_utils import create_loss_dict
from metrics.metrics_utils import normalize_metrics, write_tensorboard
from dataloader.get_dataloader import get_data_from_loader
from model_utils.eval_utils import get_losses


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def train_one_step(model, inputs, labels, criterion,
                   optimizer, lr_scheduler, writer, config,
                   tb_step_writer, scaler):
    model.train()
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    if scaler is not None:  # use AMP
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            losses, loss = get_losses(config, outputs, labels, criterion)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(inputs)
        losses, loss = get_losses(config, outputs, labels, criterion)
        loss.backward()
        optimizer.step()

    losses = create_loss_dict(config, losses)
    metrics = get_metrics(outputs, labels,
                          config['eval_metric_train']['name'])
    return outputs, losses, metrics


def train_one_epoch(model, criterion,
                    train_loader, device, epoch,
                    optimizer, lr_scheduler,
                    running_metric_train, running_loss_train,
                    writer, config, scaler):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = get_data_from_loader(data, config, device)
        outputs, loss, metrics = train_one_step(model,
                                                inputs,
                                                labels,
                                                criterion,
                                                optimizer,
                                                lr_scheduler,
                                                writer,
                                                config,
                                                tb_step_writer=i,
                                                scaler=scaler)

        running_metric_train = increment_metrics(running_metric_train,
                                                 metrics,
                                                 config=config)
        running_loss_train = increment_metrics(running_loss_train, loss)
    if lr_scheduler is not False:
        lr_scheduler.step()
    running_metric_train = normalize_metrics(running_metric_train,
                                             len(train_loader))
    running_loss_train = normalize_metrics(running_loss_train,
                                           len(train_loader))

    write_tensorboard(running_loss_train,
                      running_metric_train,
                      writer, epoch, 'train')
