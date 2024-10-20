from monai.networks import nets
from torch import nn
import torch
import os
import miacag.models.vision_transformer as vit
from collections import OrderedDict
from functools import partial

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def getPretrainedWeights(config, model, device):
    if config['model']['pretrained'] in [True, "True"]:
        if config['model']['backbone'] not in [
            'debug_3d', "vit_base_patch16_224", "vit_large_patch16_224",
            "vit_small_patch16_224", 'dinov2_vits14', 'vit_small', 'vit_large', 'vit_huge', 'vit_giant', 'vit_base', 'vit_large_3d']:
          #  if torch.distributed.get_rank() == 0:
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
            elif config['model']['backbone'] in ["vit_large_3d"]:

                loaded_model = loaded_model["encoder"]
                loaded_model = {k.replace('module.', ''): v for k, v in loaded_model.items()}
                loaded_model = {k.replace('backbone.', ''): v for k, v in loaded_model.items()}
                for k, v in model.state_dict().items():
                    if k not in loaded_model:
                        print(f'key "{k}" could not be found in loaded state dict')
                    elif loaded_model[k].shape != v.shape:
                        print(f'key "{k}" is of different shape in model and loaded state dict')
                        loaded_model[k] = v
                model.load_state_dict(loaded_model, strict=False)


            else:
                model.load_state_dict(loaded_model)
                
        else:
            if config['model']['backbone'] in ['dinov2_vits14', 'vit_small', 'vit_large', 'vit_huge', 'vit_giant', 'vit_base']:
                if config['model']['pretrain_type'] =='ijepa':
                    loaded_model = torch.load(
                            os.path.join(config['model']['pretrain_model'], 'model.pt'),
                            map_location=device)
                    if config['model']['backbone'] in ['vit_small', 'vit_large', 'vit_huge', 'vit_giant', 'vit_base']:
                        if config['loaders']['mode'] != 'testing':
                            loaded_model = {k.replace("module.", ""): v for k, v in loaded_model['target_encoder'].items()}
                    model.load_state_dict(loaded_model)
                elif config['model']['pretrain_type'] in ['mae_ct']:
                    # if mae-ct
                    loaded_model = torch.load(
                            os.path.join(config['model']['pretrain_model'], 'model.pt'), map_location=torch.device("cpu"))
                    loaded_model["pos_embed"] = torch.concat([torch.zeros(1, 1, model.embed_dim), loaded_model["pos_embed"]], dim=1)
                    model.head_drop=Identity()
                    model.head=Identity()
                    model.load_state_dict(loaded_model)
                    model.to(device)
                elif config['model']['pretrain_type'] in ['None']:
                    print('no pretraining')
                    
                elif config['model']['pretrain_type'] in ['supervised']:
                    model_path = os.path.join(
                                    config['model']['pretrain_model'],
                                    'model.pt')
                    loaded_model = torch.load(model_path, map_location=device)
                    loaded_model["pos_embed"] = loaded_model["pos_embed"][:,:-1,:] # remove class token
                    #model.head=Identity()
                    #model.head.weight.data = loaded_model["head.weight"]
                    for key in ["cls_token", "head.weight", "head.bias"]:
                        if key in loaded_model:
                            del loaded_model[key]
                    model.load_state_dict(loaded_model)
                else:
                    raise ValueError('not implemented')
            else:
                raise ValueError('not implemented')
                # model = load_from_checkpoint(
                #     model,
                #     os.path.join(config['model']['pretrain_model'], 'model.pt'),
                #     config,
                #     model_key='model|module')
            # loaded_model = torch.load(config['model']['pretrain_model'],
            #                           map_location=device)
            # remove all values from loaded_model dict starting with decoder
            # loaded_model = {k:v for k,v in loaded_model['model'].items() if not k.startswith("decoder")}
 
    elif config['model']['pretrained'] in [False, "False", "None", None]:
    #    if torch.distributed.get_rank() == 0:
        print('no pretraining is used')
        # test if it is a string
    elif type(config['model']['pretrained']) == str:
        if config['model']['backbone'] in ["vit_large_3d"]:
                print('laoading pretrained model from checkpoint')
                model_path = os.path.join(
                                config['model']['pretrained'])
                loaded_model = torch.load(
                        model_path,                    map_location=device)

                loaded_model = loaded_model["encoder"]
                loaded_model = {k.replace('module.', ''): v for k, v in loaded_model.items()}
                loaded_model = {k.replace('backbone.', ''): v for k, v in loaded_model.items()}
                for k, v in model.state_dict().items():
                    if k not in loaded_model:
                        print(f'key "{k}" could not be found in loaded state dict')
                    elif loaded_model[k].shape != v.shape:
                        print(f'key "{k}" is of different shape in model and loaded state dict')
                        loaded_model[k] = v
                model.load_state_dict(loaded_model, strict=False)
        else:
            raise ValueError('not implemented')
            dirname = os.path.dirname(__file__)
            model_path = os.path.join(
                
                            config['model']['pretrain_model'],
                            'model.pt')
            loaded_model = torch.load(
                    model_path,
                    map_location=device)

            model.load_state_dict(loaded_model)
            
            loaded_model = {k.replace('encoder.', ''): v for k, v in loaded_model.items()}
            
    else:
        raise ValueError('not implemented')           

    if config['model']['checkpoint'] != False:

        if config['model']['pretrain_model'] in  ['None', None, False, "False"]:
            pass
        else:
            if config['cpu'] == 'True':
                pretrained_dict = torch.load(config['model']['checkpoint'], map_location='cpu')
            else:

                pretrained_dict = torch.load(config['model']['checkpoint'], map_location='cuda:{}'.format(os.environ['LOCAL_RANK']))
                

            # Load your current model's state dict
            model_dict = model.state_dict()

            # Filter out unnecessary keys from the pre-trained state dict
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

            # Update your current model's state dict with the filtered pre-trained state dict
            model_dict.update(pretrained_dict)

            # Load the updated state dict into your current model
            model.load_state_dict(model_dict)
    


    



    return model


def get_encoder(config, device):


    if config['loaders']['mode'] != 'testing':
        pretrained = config['model']['pretrained']
    else:
        pretrained = False
    # if config['model']['pretrained'] == "False":
    #     pretrained = False
    # else:
    #     pretrained = config['model']['pretrained']

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
        if config['task_type'] == 'mil_classification':
            model.avgpool = nn.Identity()
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            model = nn.Sequential(*list(model.children())[:-2])
        else:
            # in_features = model.fc.in_features
            # model = nn.Sequential(*list(model.children())[:-2])
            model.avgpool = nn.Identity()
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            model = nn.Sequential(*list(model.children())[:-2])
    elif config['model']['backbone'] == 'r2plus1_18_tiny':
        model = nets.torchvision_fc.models.video.resnet.r2plus1d_18(
            pretrained=False)
        if config['loaders']['mode'] != 'testing':
            model = getPretrainedWeights(config, model, device)
        if config['task_type'] == 'mil_classification':
            model.avgpool = nn.Identity()
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            model = nn.Sequential(*list(model.children())[:-7][:-1]) #nn.Sequential(*list(model.children())[:-7])
            in_features = 64
            print('not implemented jet')
        else:
            in_features = model.fc.in_features
            model = nn.Sequential(*list(model.children())[:-2])
            #print('here')
        config['model']['backbone'] = 'r2plus1_18'
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
    elif config['model']['backbone'] == 'swin_s':
        model = nets.torchvision_fc.models.video.swin3d_s()
        in_features = model.norm.normalized_shape[0]

     #   model.avgpool = Identity()
        model.head =torch.nn.Identity()
        
    elif config['model']['backbone'] == 'swin_tiny':
        model = nets.torchvision_fc.models.video.swin3d_t()
        in_features = model.norm.normalized_shape[0]

     #   model.avgpool = Identity()
        model.head =torch.nn.Identity()
      #  model.norm = torch.nn.Identity()
    elif config['model']['backbone'] in ["mvit_base_16x4", 'mvit_base_32x3']:
     #   import pytorchvideo.models.vision_transformers as VT
        # model = VT.create_multiscale_vision_transformers(
        #     spatial_size=(config['loaders']['Crop_height'],
        #                   config['loaders']['Crop_width']),
        #     temporal_size=config['loaders']['Crop_depth'],
        #     embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
        #     atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
        #     pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
        #     pool_kv_stride_size=None,
        #     pool_kv_stride_adaptive=[1, 8, 8],
        #     pool_kvq_kernel=[3, 3, 3])
        model = nets.torchvision_fc.models.video.mvit_v1_b()

        if config['loaders']['mode'] != 'testing':
            model = getPretrainedWeights(config, model, device)
        in_features = model.norm.normalized_shape[0]
        model.head.proj = Identity()
        model.head =torch.nn.Identity()
        
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
            
    elif config['model']['backbone'] in ['vit_small_patch16',
            'vit_base_patch16', 'vit_large_patch16']:
        import miacag.models.mae_st_util.models_vit as models_vit
        config['num_frames'] = config['loaders']['Crop_depth']
        model = models_vit.__dict__[config['model']['backbone']](**config)
        in_features = model.head.in_features
        model.head = Identity()
      #  print('not implemented jet'    )
        
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
    elif config['model']['backbone'] == 'dinov2_vits14':
        model = vit_small(
                patch_size=14,
                img_size=config['loaders']['Resize_height'],
                init_values=1.0,
                block_chunks=0)
        model = getPretrainedWeights(config, model, device)
        in_features = model.norm.normalized_shape[0]
    
    elif config['model']['backbone'] in ['vit_small',
                                         'vit_base',
                                         'vit_large',
                                         'vit_huge',
                                         'vit_giant']:
        if config['model']['backbone'] == 'vit_base':
           # if config['model']['pretrain_type'] == 'mae_ct':
            from timm.models import vit_base_patch16_224
            model = vit_base_patch16_224(global_pool="token")
        #     else:
        #         pass
        #         # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        #         model = DinoVisionTransformer(
        #         patch_size=config['mask']['patch_size'],
        #         embed_dim=768,
        #         depth=12,
        #         num_heads=12,
        #         mlp_ratio=4,
        #         block_fn=partial(Block, attn_class=MemEffAttention),
        #    )
        elif config['model']['backbone'] == 'vit_small':
            
            if config['model']['pretrain_type'] == 'supervised':
            #     from timm.models import vit_small_patch16_224
            #     import timm
            #   #  model = timm.create_model('vit_small_patch16_224.augreg_in21k', pretrained=False)
              #  from dinov2.models.vision_transformer import DinoVisionTransformer, vit_small
        #    model = vit_small()
                model = vit.__dict__[config['model']['backbone']](
                    img_size=[config['loaders']['Resize_height']],
                    patch_size=config['mask']['patch_size'],
                    embed_dim=config['model']['embed_dim'])
#                model = vit_small_patch16_224(embed_dim=384)
            else:
                model = vit.__dict__[config['model']['backbone']](
                    img_size=[config['loaders']['Resize_height']],
                    patch_size=config['mask']['patch_size'],
                    embed_dim=config['model']['embed_dim'])
        else:
            model = vit.__dict__[config['model']['backbone']](
                    img_size=[config['loaders']['Resize_height']],
                    patch_size=config['mask']['patch_size'])
        #if config['model']['pretrain_type'] == 'dinov2':
          #  model = torch.load("/home/alatar/miacag/dinov2_base/model.pt")
        #else:
        model = getPretrainedWeights(config, model, device)
        if config['model']['pretrain_type'] == 'supervised':
          #  in_features = model.head.in_features
            model.head = Identity()
            model.head_drop = Identity()
        in_features = model.norm.normalized_shape[0]
        
    
    elif config['model']['backbone'] == 'vit_tiny_3d':
        from miacag.models.vision_transformer3d import vit_tiny
        model = vit_tiny(patch_size=16, img_size=224, num_frames=config['loaders']['Crop_depth'] )
        in_features = model.norm.normalized_shape[0]
        
    elif config['model']['backbone'] == 'vit_large_3d':
        from miacag.models.vision_transformer3d import vit_large
        from miacag.models.attention_pooler import AttentivePooler
        model = vit_large(patch_size=16, img_size=224, num_frames=config['loaders']['Crop_depth'] )
        model = getPretrainedWeights(config, model, device)
        
        
        in_features = model.norm.normalized_shape[0]
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
        "vit_small_patch16_224",
        "vit_small_patch16",
        "vit_base_patch16",
        "vit_large_patch16",
        "vit_huge_patch14",
        "vit_tiny_3d",
        "vit_large_3d",
        "swin_s",
        "swin_tiny",
        ]
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
