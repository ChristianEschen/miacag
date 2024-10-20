from enum import unique
from torch import nn
from miacag.models.mlps import prediction_MLP, projection_MLP
from miacag.models.get_encoder import get_encoder, modelsRequiredPermute
import torch.nn.functional as F
import torch
import numpy as np
from miacag.models.fds import FDS
def getCountsLoss(losses):
    count_regression = 0
    count_classification = 0
    for loss_u in losses:
        if loss_u in ['CE']:
            count_classification += 1
        elif loss_u in ['MSE', '_L1', 'L1smooth']:
            count_regression += 1
        else:
            raise ValueError('this loss is not implementeed:', loss_u)
    return count_regression, count_classification


def get_num_class_for_loss_group(losses, loss_group, num_classes):
    for loss in range(0, len(losses)):
        if losses[loss] == loss_group:
            return num_classes[loss]


def unique_counts(config, remove_total=False):
    if remove_total:
        if 'total' in config['loss']['name']:
            config['loss']['name'].remove('total')
    loss_uniques, count = np.unique(np.array(
        config['loss']['name']), return_counts=True)
    count = count.tolist()
    loss_uniques = loss_uniques.tolist()
    return loss_uniques, count


def get_index_for_labels_names_in_loss_group(config, group_name_unique):
    # store indexes of labels_names in a dict of dict
    indexes = {}
    indexes['loss_group'] = [[] for i in range(len(group_name_unique))]
    for group_idx, group_name in enumerate(group_name_unique):
       # idx_count = 0
        for idx, label_name in enumerate(config['labels_names']):
            if label_name.partition("_")[0] == group_name:
                indexes['loss_group'][group_idx].append(idx)
    return indexes


def get_loss_names_groups(config):
    group_name_unique,  indexes, counts = get_group_names(config)
   # group_name_unique = ["dureation", "duration", "duration"]#config['loss']['name']
  #  indexes = [0, 1, 2]
   # counts= [1, 1, 1]
    
    loss_group_names = []
    loss_lambda_weights = []
    for idx_c, index in enumerate(indexes):
        loss_group_names.append(
            config['loss']['name'][index] + '_' + group_name_unique[idx_c])
        loss_lambda_weights.append(config['loss']['lambda_weights'][index])
    indexes = get_index_for_labels_names_in_loss_group(config,
                                                       group_name_unique)
    return loss_group_names, counts, indexes, loss_lambda_weights


def get_group_names(config):
    if config['loss']['name'][0].startswith(tuple(['NNL'])):
        group_names = [i.partition("_")[0] for i in config['labels_names']]
        index = [i for i in range(len(group_names))]
        group_names_uniques = group_names
        count = [1 for i in range(len(group_names))]
        return group_names_uniques, index, count
    else:
        group_names = [i.partition("_")[0] for i in config['labels_names']]

        group_names_uniques, index, count = np.unique(np.array(
            group_names), return_counts=True, return_index=True)
        group_names_uniques = group_names_uniques.tolist()
        return group_names_uniques, index.tolist(), count.tolist()


def maybePermuteInput(x, config):
    model_list = modelsRequiredPermute()
    if config['model']['backbone'] in model_list:
        x = x.permute(0, 1, 4, 2, 3)
        return x
    else:
        return x


def maybePackPathway(frames, config):
    if config['model']['backbone'] == 'slowfast8x8':
        fast_pathway = frames
        alpha = 4
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long())
        frame_list = [slow_pathway, fast_pathway]
        return frame_list
    else:
        return frames


def get_final_layer(config,  device, in_features):
    fcs = []
    c = 0
    for head in range(0, len(config['labels_names'])):
        if config['loss']['name'][c] in ['CE']:
            fcs.append(
                nn.Linear(
                    in_features,
                    config['model']['num_classes']).to(device))
        elif config['loss']['name'][c] in ['MSE', '_L1']:
            fcs.append(
                nn.Sequential(
                    nn.Linear(
                        in_features, 1).to(device),
                    nn.Sigmoid()))
        else:
            raise ValueError('loss not implemented')
        c += 1
    return fcs


def read_tabular_data(config, checkpoint_tab_path, submodel):
    import os
            # and do it for tabular
    if len(config['loaders']['tabular_data_names']) > 0:
        if config['cpu'] == 'True':
            pretrained_dict = torch.load(checkpoint_tab_path, map_location='cpu')
        else:

            pretrained_dict = torch.load(checkpoint_tab_path, map_location='cuda:{}'.format(os.environ['LOCAL_RANK']))
            

        # Load your current model's state dict
        model_dict = submodel.state_dict()

        # Filter out unnecessary keys from the pre-trained state dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

        # Update your current model's state dict with the filtered pre-trained state dict
        model_dict.update(pretrained_dict)

        # Load the updated state dict into your current model
        submodel.load_state_dict(model_dict)
    return submodel

class EncoderModel(nn.Module):
    def __init__(self, config, device):
        super(EncoderModel, self).__init__()
        if not config['loaders']['only_tabular']:

            self.encoder, self.in_features = get_encoder(config, device)
        
    def forward(self, x):
        x = maybePermuteInput(x, self.config)
        z = self.encoder(x)
        return z


class ImageToScalarModel(EncoderModel):
    def __init__(self, config, device):
        super(ImageToScalarModel, self).__init__(config, device)
        
    
        
        self.config = config
        if self.config['loaders']['only_tabular']:
            self.in_features = 0

        if len(self.config['loaders']['tabular_data_names'])>0:
            self.tab_feature = self.config['model']['tabular_features']
        else:
            self.tab_feature = 0
        if len(self.config['loaders']['tabular_data_names'])>0:
            
            self.embeddings = nn.ModuleDict()
            embedding_dims = [10 if i!= 0 else 0 for i in config['loaders']['tabular_data_names_embed_dim']]
            embedding_dims = [min(50, i//2) for i in config['loaders']['tabular_data_names_embed_dim']]
            for i in range(0, len(config['loaders']['tabular_data_names'])):
                if config['loaders']['tabular_data_names_one_hot'][i] == 1:  # Embedding
                    self.embeddings[config['loaders']['tabular_data_names'][i]] = torch.nn.Sequential(
                        nn.Embedding(config['loaders']['tabular_data_names_embed_dim'][i], embedding_dims[i]), 
                        nn.BatchNorm1d(embedding_dims[i])
                    ).to(device)



            self.num_indicator = [1 - x for x in self.config['loaders']['tabular_data_names_one_hot']]
            self.mask_tensor = torch.tensor(self.num_indicator, dtype=bool, device=device)

            self.total_tab_features = sum(self.num_indicator) + sum(embedding_dims)
            self.layer_norm_func = nn.BatchNorm1d(sum(self.num_indicator))
            if not self.config['loaders']['only_tabular']:
                self.layer_norm_func_img = nn.BatchNorm1d(self.in_features)
            self.tabular_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.total_tab_features, self.tab_feature),
                nn.Dropout(0.2),
                torch.nn.ReLU(),
                nn.BatchNorm1d(self.tab_feature),
                torch.nn.Linear(self.tab_feature, self.tab_feature),
                nn.Dropout(0.2),
                torch.nn.ReLU(),
            ).to(device)
            self.layer_norm_before_tabular = nn.BatchNorm1d(self.tab_feature).to(device)

        if len(self.config['loaders']['tabular_data_names'])>0:
            if self.config['model']['checkpoint_tab'] != False:
            
                self.embeddings = read_tabular_data(config, config['model']['checkpoint_tab'], self.embeddings)
                self.tabular_mlp = read_tabular_data(config, config['model']['checkpoint_tab'], self.tabular_mlp)
                self.layer_norm_func = read_tabular_data(config, config['model']['checkpoint_tab'], self.layer_norm_func)
                
        
        if self.config['loaders']['only_tabular']:
            if len(self.config['loaders']['tabular_data_names'])>0:
                self.drop_out = nn.Dropout(0.5)
        #         if self.config['model']['aggregation'] == 'attention':
        #             self.att_pool = AttentiveClassifier(
        #                     embed_dim=self.encoder.embed_dim,
        #                     num_heads=self.encoder.num_heads,
        #                     depth=1,
        #                     num_classes=count_loss,
        #                 ).to(device)
        #         else:
        #             pass
        
        
        self.keys_total = config['loaders']['tabular_data_names']
        
        self.numeric_keys = [self.keys_total[i] for i in range(len(self.keys_total)) if config['loaders']['tabular_data_names_one_hot'][i] == 0]

        self.dimension = config['model']['dimension']
        self.fcs = nn.ModuleList()
        for loss_count_idx, loss_type in enumerate(self.config['loss']['groups_names']):

                
           # count_loss = counts[loss_count_idx]
            count_loss = self.config['loss']['groups_counts'][loss_count_idx]
            if loss_type.startswith(tuple(['CE'])):
                self.fcs.append(nn.Linear(
                        self.in_features,
                        config['model']['num_classes'][loss_count_idx]).to(device))
                
            elif loss_type.startswith(tuple(['NNL'])):
                self.fcs.append(
                        nn.Sequential(
                            nn.BatchNorm1d(self.tab_feature + self.in_features),
                            nn.Linear(
                                self.tab_feature + self.in_features, self.in_features+ self.tab_feature).to(device),
                            torch.nn.Dropout(0.2),
                            nn.ReLU(),
                            nn.BatchNorm1d(self.in_features + self.tab_feature),
                            nn.Linear(
                                self.in_features+ self.tab_feature, self.in_features+ self.tab_feature).to(device),
                            torch.nn.Dropout(0.1),
                            nn.ReLU(),
                            nn.BatchNorm1d(self.in_features + self.tab_feature),
                            nn.Linear(
                                self.in_features+ self.tab_feature,
                                config['model']['num_classes'][loss_count_idx]).to(device),
                            ))
                

            elif loss_type.startswith(tuple(['BCE_multilabel'])):
                self.fcs.append(
                    nn.Sequential(
                        nn.Linear(
                            self.in_features, self.in_features).to(device),
                        nn.ReLU(),
                        nn.Linear(
                            self.in_features, count_loss)
                        ).to(device),
                      #  nn.ReLU()
                        )
            # test if loss_type startswith three conditions
            
            elif loss_type.startswith(tuple(['MSE', '_L1', 'L1smooth', 'wfocall1'])):
                if config['model']['aggregation'] == 'mean':   
                    self.fcs.append(
                        nn.Sequential(
                            nn.Linear(
                                self.in_features, self.in_features).to(device),
                            nn.ReLU(),
                            nn.Linear(
                                self.in_features,
                                count_loss).to(device),
                            nn.ReLU(),

                            ))
                elif config['model']['aggregation'] == 'cross_attention':
                    from miacag.models.attention_pooler import AttentivePooler, AttentiveClassifier
                    self.att_pool = AttentiveClassifier(
                        embed_dim=self.encoder.embed_dim,
                        num_heads=self.encoder.num_heads,
                        depth=1,
                        num_classes=count_loss,
                    ).to(device)


                else:
                    ValueError('aggregation not implemented', config['models']['aggregation'])

            else:
                raise ValueError('loss not implemented')
        if self.config['model']['fds'] == True:

            if self.config['model']['fds'] == True:
                start_smooth = self.config['model']['start_smooth']
                self.FDS = FDS(
                    feature_dim=self.in_features, bucket_num=100, bucket_start=3,
                    start_update=0, start_smooth=start_smooth, kernel="gaussian", ks=9, sigma=1, momentum=0.9
                )
        
            self.fds =self.config['model']['fds'] 
            self.start_smooth = start_smooth
    def tabular_forward(self, tabular_data):
        embedded_features = []
        counter = 0
        for key, flag in zip(self.config['loaders']['tabular_data_names'], self.config['loaders']['tabular_data_names_one_hot']):
            if flag == 1:  # Embedding
                # make copy of the tensor tabular_data
            #    tabular_data_cat = torch.clone(tabular_data[:,counter])
                embedded = self.embeddings[key](tabular_data[:,counter].long())  # Remove the extra dimension if necessary
                embedded_features.append(embedded)

            counter += 1
                            # Directly use the normalized numeric data
        tabular_data_numeric = self.layer_norm_func(tabular_data[:,self.mask_tensor])
        # if len(embedded_features) > 0:
        #     if len(embedded_features[0].shape) ==1:
        #         embedded_features[0] = embedded_features[0].unsqueeze(0)
        embedded_features.append(tabular_data_numeric)
        return torch.cat(embedded_features, dim=1)
        
    def forward(self, x = None, tabular_data = None, targets = None, epoch = None):
        ## TODO implement tabular only setting
        if not self.config['loaders']['only_tabular']:
          #  if self.config['loaders']['val_method']['saliency'] == False:
            x = maybePermuteInput(x, self.config)
            
            p = self.encoder(x)
            if self.config['model']['aggregation'] in ['max','mean']:
                if self.dimension in ['3D', '2D+T']:
                    if self.config['model']['backbone'] not in [
                        "mvit_base_16x4", "mvit_base_32x3", "vit_base_patch16_224",
                        "vit_small_patch16_224", "vit_large_patch16_224",
                        "vit_base_patch16", "vit_small_patch16", "vit_large_patch16",
                        "vit_huge_patch14", "swin_s", "swin_tiny"]:
                        
                        if self.config['model']['backbone'] in [
                        "vit_tiny_3d", "vit_small_3d", "vit_base_3d", "vit_large_3d"]:
                            p = p.mean(dim=(-2))
                        else:
                            p = p.mean(dim=(-3, -2, -1))
                    else:
                        pass
                elif self.dimension == 'tabular':
                    p = p
                else:
                    p = p.mean(dim=(-2, -1))
                    
                
                # apply the FDS
                encoding_s = p

                if self.training and self.fds:
                    if epoch >= self.start_smooth:
                        encoding_s = self.FDS.smooth(encoding_s, targets, epoch, self.config)
                else:
                    encoding_s = None

                ps = []
                if len(self.config['loaders']['tabular_data_names'])>0:
                    encode_num_and_cat_feat = self.tabular_forward(tabular_data)
                    tabular_features = self.tabular_mlp(encode_num_and_cat_feat)
                    ##### Gradient blending #####
                    # tabular_features = self.layer_norm_before_tabular(tabular_features)

                    # tabular_fc_out = self.tabular_fc(tabular_features)
                    # vis_fc_out = self.vis_fc(p)
                    #############################
                    if self.config['loaders']['mode'] == 'testing':
                        tabular_features = torch.cat([tabular_features] * x.shape[0], dim=0)
                    if not self.config['loaders']['only_tabular']:
                        p = self.layer_norm_func_img(p)
                    p = torch.concat((p, tabular_features), dim=1)  # Combine the features from video and tabular data
                    if self.config['loaders']['only_tabular']:
                        if len(self.config['tabular_data_names'])>0:
                            p = self.drop_out(p)

                for fc in self.fcs:
                    features = fc(p)
                    ps.append(features)
                

                
            else:
                ps = [self.att_pool(p)]
            
            if self.config['loaders']['val_method']['saliency'] == True:
                ps = torch.cat(ps, dim=1)
                return ps
        else:
            encoding_s = None
            encode_num_and_cat_feat = self.tabular_forward(tabular_data)
            tabular_features = self.tabular_mlp(encode_num_and_cat_feat)
            if self.config['loaders']['mode'] == 'testing':
                tabular_features = torch.cat([tabular_features] * 1, dim=0)
            p = tabular_features
            ps = []
            for fc in self.fcs:
                    features = fc(p)
                    ps.append(features)
        if len(self.config['loaders']['tabular_data_names'])>0:
            if len(self.config['labels_names'])>1:
                return (ps[0],  tabular_fc_out, vis_fc_out)
            else:
                return [ps[0], encoding_s]
        else:
            return (ps, encoding_s)

   