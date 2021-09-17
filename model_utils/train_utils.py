import torch
import numpy as np
import random
from metrics.metrics_utils import get_metrics, increment_metrics
from metrics.metrics_utils import create_loss_dict
from metrics.metrics_utils import normalize_metrics, write_tensorboard
from dataloader.get_dataloader import get_data_from_loader
from model_utils.eval_utils import get_losses
from configs.config import save_config
from metrics.metrics_utils import flatten, \
    unroll_list_in_dict
import os


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
                                             config,
                                             len(train_loader.dataset.data))
    running_loss_train = normalize_metrics(running_loss_train,
                                           config,
                                           len(train_loader.dataset.data))

    write_tensorboard(running_loss_train,
                      running_metric_train,
                      writer, epoch, 'train')


def test_best_loss(best_val_loss, best_val_epoch, val_loss, epoch):
    if best_val_loss is None or best_val_loss > val_loss:
        best_val_loss, best_val_epoch = val_loss, epoch
    return best_val_loss, best_val_epoch


def early_stopping(best_val_loss, best_val_epoch,
                   val_loss, epoch, max_stagnation):
    early_stop = False

    best_val_loss, best_val_epoch = test_best_loss(best_val_loss,
                                                   best_val_epoch,
                                                   val_loss,
                                                   epoch)

    if best_val_epoch < epoch - max_stagnation:
        # nothing is improving for a while
        early_stop = True

    return early_stop, best_val_loss, best_val_epoch


def write_log_file(config, writer):
    f = open(config['logfile'], "w")
    f.write(writer.log_dir)
    f.close()


def get_device(config):
    if config["cpu"] == "False":
        if config['loaders']['mode'] == 'training':
            device = "cuda:{}".format(config['local_rank'])
        else:
            device = "cuda:0"
    else:
        device = 'cpu'
    device = torch.device(device)
    return device


def save_model(model, writer, config):
    model_file_path = os.path.join(writer.log_dir, 'model.pt')
    model_encoder_file_path = os.path.join(
            writer.log_dir, 'model_encoder.pt')

    if config["cpu"] == "False":
        torch.save(model.module.state_dict(), model_file_path)
    else:
        torch.save(model.state_dict(), model_file_path)

    if config["task_type"] in ["representation_learning"]:
        if config["cpu"] == "False":
            torch.save(model.module.encoder.state_dict(), model_encoder_file_path)
        else:
            torch.save(model.encoder.state_dict(), model_encoder_file_path)


def saver(metric_dict_val, writer, config):
    # prepare dicts by flattening
    config = unroll_list_in_dict(flatten(config))
    metric_dict_val = {str(key)+'/val': val
                       for key, val in metric_dict_val.items()}
    # save config
    config.update(metric_dict_val)
    save_config(writer, config)
    # save tensorboard
    writer.add_hparams(config, metric_dict=metric_dict_val)
    writer.flush()
    writer.close()
    write_log_file(config, writer)
