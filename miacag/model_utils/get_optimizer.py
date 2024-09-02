import os
import torch
from miacag.model_utils.scheduler import WarmupMultiStepLR
import math
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer(config, model, len_train):
    if config['use_DDP'] == 'False':
        os.environ['WORLD_SIZE'] = '1'
    if config['optimizer']['type'] == 'adam':

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['optimizer']['learning_rate'],
            weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['learning_rate'],
            weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['type'] == 'sgd':
        lr_image_encoder = 0.02
        lr_tabular_encoder = 0.002

        # Create parameter groups
        optimizer = torch.optim.Adam([
            {'params': model.module.encoder.parameters(), 'lr': lr_image_encoder, 'momentum':config['optimizer']['momentum'], 'weight_decay': config['optimizer']['momentum']},
            {'params': model.module.layer_norm_func_img.parameters(), 'lr': lr_image_encoder, 'momentum':config['optimizer']['momentum'], 'weight_decay': config['optimizer']['momentum']},
            {'params': model.module.tabular_mlp.parameters(), 'lr': lr_tabular_encoder, 'momentum':config['optimizer']['momentum'], 'weight_decay': config['optimizer']['momentum']},
            {'params': model.module.embeddings.parameters(), 'lr': lr_tabular_encoder, 'momentum':config['optimizer']['momentum'], 'weight_decay': config['optimizer']['momentum']},
            {'params': model.module.layer_norm_func.parameters(), 'lr': lr_tabular_encoder, 'momentum':config['optimizer']['momentum'], 'weight_decay': config['optimizer']['momentum']},
            {'params': model.module.fcs.parameters(), 'lr': lr_image_encoder, 'momentum':config['optimizer']['momentum'], 'weight_decay': config['optimizer']['momentum']},  # or a different learning rate if needed
        ])
        # optimizer = torch.optim.SGD(
        #     model.parameters(),
        #     lr=config['optimizer']['learning_rate'],
        #     momentum=config['optimizer']['momentum'],
        #     weight_decay=config['optimizer']
        #                         ['weight_decay'])
    # Set learning rate scheduler
    if config['lr_scheduler']['type'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_scheduler']['steps_for_drop'],
            gamma=0.1)
    elif config['lr_scheduler']['type'] == 'OneCycleLR':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['optimizer']['learning_rate'],
                                                           steps_per_epoch=config['lr_scheduler']["steps_per_epoch"], epochs=config['trainer']['epochs'])
    elif config['lr_scheduler']['type'] in ['MultiStepLR', 'poly', 'cos', "warmup_const", "coswarm"]:
        if config['lr_scheduler']['type'] == 'poly':
            print('to be implemented')
            lr_scheduler = PolynomialLRDecay(
                optimizer,
                max_decay_steps=config['trainer']['epochs'],
                end_learning_rate=config['lr_scheduler']['end_lr'],
                power=config['lr_scheduler']['power'])

        elif config['lr_scheduler']['type'] == 'MultiStepLR':
            warmup_iters = config['lr_scheduler']['lr_warmup_epochs'] \
                * len_train
            lr_milestones = [
                len_train * m for m in config['lr_scheduler']['milestones']]
            lr_scheduler = WarmupMultiStepLR(
                    optimizer,
                    milestones=lr_milestones,
                    gamma=config['lr_scheduler']['gamma'],
                    warmup_iters=warmup_iters,
                    warmup_factor=1e-5,
                )
        elif config['lr_scheduler']['type'] == 'warmup_const':
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=config['lr_scheduler']['lr_warmup'], total_iters=config['lr_scheduler']['nr_warmup_epochs'])
            scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=config['optimizer']['learning_rate'], total_iters=config['lr_scheduler']['nr_warmup_epochs'])
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
        elif config['lr_scheduler']['type'] == 'cos':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config['trainer']['epochs'])
        elif config['lr_scheduler']['type'] == 'coswarm':
            
            total_epochs = config['trainer']['epochs']
            warmup_epochs = config['lr_scheduler']['nr_warmup_epochs']
            def lr_lambda(epoch):
                return epoch / warmup_epochs if epoch < warmup_epochs else 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

            lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        lr_scheduler = False

    # warmup_steps = 200
    # base_lr = 0.000001
    # lambda1 = lambda step: step / warmup_steps if step < warmup_steps else 1
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return optimizer, lr_scheduler

# def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
#                         max_iter=100, power=0.9):
#     """Polynomial decay of learning rate
#         :param init_lr is base learning rate
#         :param iter is a current iteration
#         :param lr_decay_iter how frequently decay occurs, default is 1
#         :param max_iter is number of maximum iterations
#         :param power is a polymomial power

#     """
#     if iter % lr_decay_iter or iter > max_iter:
#         return optimizer

#     lr = init_lr*(1 - iter/max_iter)**power
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#     return lr
