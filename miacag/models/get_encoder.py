from monai.networks import nets
from torch import nn
import torch
import os
from timm.models import create_model
from models import modeling_finetune
from collections import OrderedDict


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def getPretrainedWeights(config, model, device):
    if config['model']['pretrained'] == "True":
        if config['model']['backbone'] not in [
            'debug_3d', "vit_base_patch16_224", "vit_large_patch16_224",
            "vit_small_patch16_224"]:
            if torch.distributed.get_rank() == 0:
                dirname = os.path.dirname(__file__)
                model_path = os.path.join(
                                dirname,
                                "torchhub",
                                config['model']['dimension'],
                                config['model']['backbone'],
                                'model.pt')
                loaded_model = torch.load(
                        model_path,
                        map_location=device)

                if config['model']['backbone'] in \
                        ['x3d_s', 'slowfast8x8', "mvit_base_16x4", 'mvit_base_32x3']:

                    model.load_state_dict(loaded_model['model_state'])
                else:
                    model.load_state_dict(loaded_model)
        else:
            model = load_from_checkpoint(
                model,
                config['model']['pretrain_model'],
                config,
                model_key='model|module')
            # loaded_model = torch.load(config['model']['pretrain_model'],
            #                           map_location=device)
            # remove all values from loaded_model dict starting with decoder
            # loaded_model = {k:v for k,v in loaded_model['model'].items() if not k.startswith("decoder")}

    elif config['model']['pretrained'] == "None":
        pass
    else:
        if torch.distributed.get_rank() == 0:
            dirname = os.path.dirname(__file__)
            model_path = os.path.join(
                            config['model']['pretrain_model'],
                            'model.pt')
            loaded_model = torch.load(
                    model_path,
                    map_location=device)

            model.load_state_dict(loaded_model)
    return model


def get_encoder(config, device):
    if config['loaders']['mode'] != 'testing':
        pretrained = config['model']['pretrained']
    else:
        pretrained = False

    # Get model
    if config['model']['backbone'] == 'r3d_18':
        model = nets.torchvision_fc.models.video.resnet.r3d_18(
            pretrained=pretrained)
        in_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-2])
        # model.fc = nn.Identity()
    elif config['model']['backbone'] == 'r2plus1_18':
        model = nets.torchvision_fc.models.video.resnet.r2plus1d_18(
            pretrained=False)
        if config['loaders']['mode'] != 'testing':
            model = getPretrainedWeights(config, model, device)
        in_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-2])
    elif config['model']['backbone'] == 'x3d_l':
        print('not implemented jet')
    elif config['model']['backbone'] == 'slowfast8x8':
        print('not impl')
        import pytorchvideo.models.slowfast as slowfast
        model = slowfast.create_slowfast()
        model = getPretrainedWeights(config, model, device)
        in_features = model.blocks[-1].proj.in_features
        model = nn.Sequential(
            *(list(model.blocks[:-1].children()) +
              list(model.blocks[-1].children())[:-2]))
    elif config['model']['backbone'] == 'x3d_s':
        import pytorchvideo.models.x3d as x3d
        model = x3d.create_x3d()  # default args creates x3d_s
        if config['loaders']['mode'] != 'testing':
            model = getPretrainedWeights(config, model, device)
        in_features = model.blocks[-1].proj.in_features
        model = nn.Sequential(
            *(list(model.blocks[:-1].children()) +
              list(model.blocks[-1].children())[:-3]))

    elif config['model']['backbone'] in ["mvit_base_16x4", 'mvit_base_32x3']:
        import pytorchvideo.models.vision_transformers as VT
        model = VT.create_multiscale_vision_transformers(
            spatial_size=(config['loaders']['Crop_height'],
                          config['loaders']['Crop_width']),
            temporal_size=config['loaders']['Crop_depth'],
            embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
            atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
            pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
            pool_kv_stride_size=None,
            pool_kv_stride_adaptive=[1, 8, 8],
            pool_kvq_kernel=[3, 3, 3])
        if config['loaders']['mode'] != 'testing':
            model = getPretrainedWeights(config, model, device)
        in_features = model.head.proj.in_features
        model.head.proj = Identity()
        
    elif config['model']['backbone'] in [
        'pretrain_videomae_small_patch16_224',
        "pretrain_videomae_base_patch16_224", "vit_base_patch16_224",
        "vit_large_patch16_224",
        "vit_small_patch16_224"]:
        model = get_model(config)
        model.head = Identity()
        in_features = model.fc_norm.normalized_shape[0]
        if config['loaders']['mode'] != 'testing':
            model = getPretrainedWeights(config, model, device)
    elif config['model']['backbone'] == 'linear':
        from miacag.models.backbone_encoders.tabular.base_encoders \
            import LinearEncoder
        model = LinearEncoder(config['model']['incomming_features'])
        in_features = config['model']['incomming_features']

    elif config['model']['backbone'] == 'mlp':
        from miacag.models.mlps import projection_MLP
        in_features = 128
        model = projection_MLP(config['model']['incomming_features'],
                               in_features)
    elif config['model']['backbone'] == 'r50':
        from torchvision.models import resnet50
        model = resnet50()
        if config['loaders']['mode'] != 'testing':
            model = getPretrainedWeights(config, model, device)
        in_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-2])
    elif config['model']['backbone'] == 'debug_3d':
        from miacag.models.cnns import debug_3d
        in_features = 16
        model = debug_3d(in_features)
        model = getPretrainedWeights(config, model, device)
        
    else:
        raise ValueError('not implemented')
    return model, in_features


def modelsRequiredPermute():
    model_list = [
        'r3d_18', 'r2plus1_18',
        'x3d_l', 'x3d_s', "mvit_base_16x4", 'mvit_base_32x3', 'slowfast8x8',
        'pretrain_videomae_base_patch16_224',
        'vit_base_patch16_224',
        "vit_large_patch16_224", 
        "vit_small_patch16_224"]
    return model_list


# from VideoMAE

def get_model(config):
    print(f"Creating model: {config['model']['backbone']}")
 #   model = create_model(
    #     config['model']['backbone'],
    #     pretrained=False,
    #     drop_path_rate=config['drop_path'],
    #     drop_block_rate=None,
    #     decoder_depth=config['decoder_depth'],
    #     use_checkpoint=config['use_checkpoint']
    # )
    model = create_model(
        config['model']['backbone'],
        pretrained=False,
        drop_path_rate=0.0,
        drop_block_rate=None)
    return model


def my_load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
        

def load_from_checkpoint(model,
                         fine_tune_path,
                         config,
                         model_key='model|module'):
    checkpoint = torch.load(fine_tune_path, map_location='cpu')

    print("Load ckpt from %s" % fine_tune_path)
    checkpoint_model = None
    for model_key in "model|module".split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # height (== width) for the checkpoint position embedding 
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(config['loaders']['Crop_depth'] // model.patch_embed.tubelet_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (config['loaders']['Crop_depth'] // model.patch_embed.tubelet_size) )** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, config['loaders']['Crop_depth'] // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, config['loaders']['Crop_depth'] // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            
    my_load_state_dict(model, checkpoint_model, prefix='')
    return model
