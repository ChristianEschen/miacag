import torch
import numpy as np
import random
from miacag.metrics.metrics_utils import get_metrics, get_losses_metric
from miacag.metrics.metrics_utils import create_loss_dict, get_loss_metric_class
from miacag.metrics.metrics_utils import normalize_metrics, write_tensorboard
from miacag.dataloader.get_dataloader import get_data_from_loader
from miacag.model_utils.eval_utils import get_loss
from miacag.configs.config import save_config
from miacag.metrics.metrics_utils import flatten, \
    unroll_list_in_dict
import os
from miacag.utils.common_utils import stack_labels, get_loss
from miacag.utils.common_utils import get_losses_class, wrap_outputs_to_dict
from miacag.models.BuildModel import ModelBuilder
from miacag.models.modules import get_loss_names_groups
from torch.nn.utils.clip_grad import clip_grad_norm
import tqdm

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_candiate_label(config, group_idx):
    label_candidate = [config['labels_names'][i] for i in group_idx]
    return label_candidate


def get_correct_candidate_index(config, candiates, label_name):
    for candidate in candiates:
        if candidate == label_name:
            return candidate


def train_one_step(model, data, criterion,
                   optimizer, lr_scheduler, writer, config,
                   running_loss_train,
                   running_metric_train,
                   epoch,
                   tb_step_writer, scaler, device, iter_minibatch):
    model.train()    
    
    if config['loaders']['tabular_data_names'] == []:
        data['tabular_data'] = None
    if config['loaders']['only_tabular']:
        data['inputs'] = None
    if config['model']['fds']:
        targets = torch.concat([data[q] for q in config['labels_names']], dim=0)

    if scaler is not None:  # use AMP
        with torch.cuda.amp.autocast():
       # with torch.autocast(device_type='cuda', dtype=torch.float16):

            outputs, _ = model(data['inputs'], data['tabular_data'], targets, epoch)
            losses, loss = get_losses_class(config,
                                            outputs,
                                            data,
                                            criterion,
                                            device)
        loss = loss / config['accum_iter']

        scaler.scale(loss).backward()
       # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        if (iter_minibatch + 1) % config['accum_iter'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    else:
        outputs, _ = model(data['inputs'], data['tabular_data'], targets, epoch)


        losses, loss = get_losses_class(config,
                                        outputs,
                                        data,
                                        criterion,
                                        device)
        loss = loss / config['accum_iter']

        loss.backward()
       # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if (iter_minibatch + 1) % config['accum_iter'] == 0:
                optimizer.step()
                optimizer.zero_grad()
    outputs = wrap_outputs_to_dict(outputs, config)
    losses = create_loss_dict(config, losses, loss)
    metrics, losses_metric = get_loss_metric_class(
        config, outputs, data, losses, running_metric_train,
        running_loss_train, criterion)
    return outputs, losses, metrics, losses_metric


def train_one_epoch(model, criterion,
                    train_loader, device, epoch,
                    optimizer, lr_scheduler,
                    running_metric_train, running_loss_train,
                    writer, config, scaler, iter_minibatch):
        # if config['loaders']['mode'] == 'testing':
   # train_loader.dataset.data = train_loader.dataset.data*400
    for i, data in enumerate(train_loader, 0):
        iter_minibatch += 1
        data = get_data_from_loader(data, config, device)
      #  if epoch >= 2:
      
        # from miacag.dataloader.get_dataloader import display_input, display_input_stats
        #from miacag.dataloader.get_dataloader import display_input
        #if i ==1:
        #    display_input(data["inputs"], config)
        #display_input_stats(data)
       # print('1 prox', data['sten_proc_1_prox_rca_transformed'])
      # print('2 midt', data['sten_proc_2_midt_rca_transformed'])
        # if epoch < 10:
        #     lr_scheduler.base_lrs[0] =lr_scheduler.base_lrs[0]*0.1
            # lr_scale = warm_up_factor * current_step
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = 0.1 * lr_scal
        outputs, loss, metrics, loss_metric = train_one_step(
            model,
            data,
            criterion,
            optimizer,
            lr_scheduler,
            writer,
            config,
            running_loss_train,
            running_metric_train,
            epoch,
            tb_step_writer=i,
            scaler=scaler,
            device=device,
            iter_minibatch=iter_minibatch)
        # running_metric_train = increment_metrics(running_metric_train,
        #                                          metrics)
       # running_loss_train = increment_metrics(running_loss_train, loss)
    
    if config['model']['fds'] and epoch >= 0:
        print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for c, data in enumerate(train_loader, 0):
                if config['loaders']['tabular_data_names'] == []:
                        data['tabular_data'] = None
                data['inputs'] = data['inputs'].cuda(non_blocking=True)
                targets = torch.concat([data[q] for q in config['labels_names']], dim=0)
                _, feature = model(data['inputs'],data['tabular_data'],targets, epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())

        encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
        model.module.FDS.update_last_epoch_stats(epoch)
        model.module.FDS.update_running_stats(encodings, labels, epoch)

    running_metric_train, metric_tb = normalize_metrics(
        metrics, device)
    running_loss_train, loss_tb = normalize_metrics(
        loss_metric, device)

    loss_tbe, metric_tb = write_tensorboard(loss_tb,
                                            metric_tb,
                                            writer,
                                            epoch,
                                            'train')
    if lr_scheduler is not False:
        lr_scheduler.step()

  #  if metrics is not None:  # Add check her

    if torch.distributed.get_rank() == 0:
        if config['lr_scheduler']['type'] == 'coswarm':
            cur_lr = lr_scheduler.optimizer.param_groups[0]['lr']
        else:
            cur_lr= lr_scheduler.get_last_lr()[0]
        print('cur lr', cur_lr)
        print('lr_scheduler', lr_scheduler)
        writer.add_scalar(
            "learning rate",
             cur_lr,  # losses[loss],
             epoch)
    return iter_minibatch


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

def get_device(config):
    if config["cpu"] == "False":
        device = "cuda:{}".format(os.environ['LOCAL_RANK'])
    else:
        device = 'cpu'
    device = torch.device(device)
    return device

def fake_saving_ddp_model_wrapper(config, model_file_path):
    device = get_device(config)
    config['datasetFingerprintFile'] = None
    config['loaders']['mode'] = 'training'
    config['loss']['groups_names'], config['loss']['groups_counts'], \
        config['loss']['group_idx'], config['groups_weights'] \
        = get_loss_names_groups(config)
    config['use_DDP'] = 'True'
    if config["cpu"] == "False":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    BuildModel = ModelBuilder(config, device)
    model = BuildModel()
    if config["cpu"] == "False":
        torch.save(model.module.state_dict(), model_file_path)
    else:
        torch.save(model.state_dict(), model_file_path)


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
    config_tensorboard = unroll_list_in_dict(flatten(config))
    metric_dict_val = {str(key)+'/val': val
                       for key, val in metric_dict_val.items()}
    # save config
    config_tensorboard.update(metric_dict_val)
    if config['loss']['name'][0] == 'NNL':
        config['cuts'] = list(config['cuts'] )

    save_config(writer, config, 'config.yaml')
    save_config(writer, metric_dict_val, 'metrics.yaml')

    # remove None values and lists
    for key in list(config_tensorboard.keys()):
        if config_tensorboard[key] is None:
            del config_tensorboard[key]
        elif isinstance(config_tensorboard[key], list):
            del config_tensorboard[key]

    # save tensorboard
    if config_tensorboard['loss_name0'] =='NNL':
        config_tensorboard['cuts'] = str(config_tensorboard['cuts'])
    writer.add_hparams(config_tensorboard, metric_dict=metric_dict_val)
    writer.flush()
    writer.close()
 