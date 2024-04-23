import torch
import os
import numpy as np

def to_dtype(data,fields, config):
    for c, label_name in enumerate(fields):
        if config['loss']['name'][c].startswith('CE'):
            data[label_name] = torch.nan_to_num(data[label_name], nan=99998)
            data[label_name] = data[label_name].long()
        elif config['loss']['name'][c] == 'BCE_multilabel':
            data[label_name] = torch.nan_to_num(data[label_name], nan=99998)
            data[label_name] = data[label_name].long()
        elif config['loss']['name'][c] in ['MSE', '_L1', 'L1smooth','wfocall1']:
            data[label_name] = data[label_name].float()
        elif config['loss']['name'][c] in ['NNL']:
            data[label_name] = data[label_name].float()
            data['event'] = data['event'].int()
        else:
            raise ValueError("model loss not implemented")
        data
    return data


def to_device(data, device, fields):
    for field in fields:
        if field.startswith('duration'):
            data['event'] = data['event'].to(device)
        data[field] = data[field].to(device)
    return data

    

def map_category(value):
    if torch.isnan(value):
        return 2.0
    elif value == 0:
        return 0.5
    elif value == 1:
        return 1.0
    elif value == 2:
        return 1.5
    elif value == None:
        return 2.0

def display_input_stats(data):
    print('rowid', data['rowid'])
    print('data shape', data['inputs'].shape)
    print('data mean', data['inputs'].mean())
    print('data std', data['inputs'].std())
    print('data max', data['inputs'].max())
    print('data min', data['inputs'].min())
    for p in range(0, data["inputs"].shape[1]):
        print('patch', p)
        print('patch mean', data['inputs'][:, p, :, :, :, :].mean())
        print('patch std', data['inputs'][:, p, :, :, :, :].std())
        print('patch max', data['inputs'][:, p, :, :, :, :].max())
        print('patch min', data['inputs'][:, p, :, :, :, :].min())
def display_input(data, config):
    # data is a torch tensor of size [bs, patches, ch, h, w, depth]
    if config['model']['dimension'] == '2D':
        counter = 0
        for i in range(0, data.shape[0]):
            for p in range(0, data.shape[1]):
                print('counter', counter)   

                counter += 1
                slice = data[i, p, :, :, :]
                # normalize slice to [0, 1]
                slice = (slice - slice.min()) / (slice.max() - slice.min())
                slice = slice.cpu().numpy()
                # cast to uint8
                slice = (slice * 255).astype(np.uint8)
                # permute slice to [h, w, ch]
                slice = np.transpose(slice, (1, 2, 0))
                
                
                # show torch tensor as image
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('TkAgg')
                plt.imshow(slice)
                # set grayscale color map
                plt.set_cmap('gray')
                plt.show()
    else:
        
        counter = 0
        for i in range(0, data.shape[0]):
            for p in range(0, data.shape[1]):
                for d in range(0, data.shape[5]):
                    print('counter', counter)   

                    counter += 1
                    slice = data[i, p, :, :, :, d]
                    # normalize slice to [0, 1]
                    slice = (slice - slice.min()) / (slice.max() - slice.min())
                    slice = slice.cpu().numpy()
                    # cast to uint8
                    slice = (slice * 255).astype(np.uint8)
                    # permute slice to [h, w, ch]
                    slice = np.transpose(slice, (1, 2, 0))
                    
                    
                    # show torch tensor as image
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use('TkAgg')
                    plt.imshow(slice)
                    # set grayscale color map
                    plt.set_cmap('gray')
                    plt.show()
    print('done')
def encode_labels_predictions_in_corner(data, categories):
    batch_size = data["inputs"].shape[0]
    # Encode categorical values into the top-left corner of each patch
    for i in range(batch_size):
        # Map the category value to the desired encoding
        encoded_value = map_category(categories[i])

        # Apply the encoded value to the top-left 2x2 pixels of each patch
        #for j in range(patches):
        data['inputs'][i, :, 0:2, 0:2, :] = encoded_value
    return data
def get_data_from_loader(data, config, device, val_phase=False):
    if config['loaders']['store_memory'] is True:
        data = {
                'inputs': data[0],
                config['labels_names']: data[1]
                }
    if config['task_type'] in ["classification", "regression", "mil_classification"]:
        # rename data["DcmPathFlatten"] to data["inputs"]
        if config["task_type"] == "mil_classification":
            data['inputs'] = data['DcmPathFlatten']
            display_input_stats(data)

       # else:
            # print('data shape', data['inputs'][0].shape)
            # print('data mean', data['inputs'][0].mean())
            # print('data std', data['inputs'][0].std())
            # print('data max', data['inputs'][0].max())
            # print('data min', data['inputs'][0].min())
        data["inputs"] = torch.tensor(data["inputs"])
        data = to_device(data, device, ['inputs'])
        
       # data = encode_labels_predictions_in_corner(data, data['labels_predictions'])
        # display_input_stats(data)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('TkAgg')
        # display_input(data['inputs'])
      #  print('data mean', data['inputs'].mean())
      #  print('data std', data['inputs'].std())
        # display each frame it consit of tensor of shape [bs, pathcies, ch, h, w, depth]
        if config['loss']['name'][0] in ['MSE', '_L1', 'L1smooth','wfocall1']: 
            data = to_device(data, device, ["weights_" + i for i in config['labels_names']])
        data = to_dtype(data, config['labels_names'], config)
        if config['loss']['name'][0] in ['MSE', '_L1', 'L1smooth','wfocall1']: 
            data = to_dtype(data, ["weights_" + i for i in config['labels_names']], config)
        data = to_device(data, device, config['labels_names'])

       # if 
        #print('data shape', data['inputs'].shape)
    elif config['task_type'] == "representation_learning":
        if val_phase is False:
            if config['loaders']['store_memory'] is False:
                data['inputs'] = data['inputs'].to(device)
                if config['model']['dimension'] == '2D':
                    data['inputs'] = torch.cat(
                        (torch.unsqueeze(inputs[::2, :, :, :], dim=1),
                            torch.unsqueeze(inputs[1::2, :, :, :], dim=1)),
                        dim=1)
                elif config['model']['dimension'] in ['3D', '2D+T']:
                    data['inputs'] = torch.cat(
                        (torch.unsqueeze(inputs[::2, :, :, :, :], dim=1),
                            torch.unsqueeze(inputs[1::2, :, :, :, :], dim=1)),
                        dim=1)
                else:
                    raise ValueError(
                            "model dimension not implemented")
            else:
                data['inputs'] = torch.cat(
                    (torch.unsqueeze(data[0][0].to(device), 1),
                     torch.unsqueeze(data[0][1].to(device), 1)),
                    dim=1)
            data['labels'] = None
        else:
            if config['loaders']['store_memory'] is False:
                data['inputs'] = data['inputs'].to(device)
                data['labels'] = data[config['labels_names']].long().to(device)
            else:
                data['inputs'] = data[0].to(device)
                data['labels'] = data[1].long().to(device)
    else:
        raise ValueError(
                "Data type is not implemented")

    return data

def get_data_from_standard_Datasets(data, config, device, val_phase):
    data = {
                'inputs': data[0],
                config['labels_names']: data[1]
                }
    if config['task_type'] == "representation_learning":
        if val_phase is False:
            data['inputs'] = torch.cat(
                        (torch.unsqueeze(data[0][0].to(device), 1),
                        torch.unsqueeze(data[0][1].to(device), 1)),
                        dim=1)


def get_dataloader_train(config):
    if config['task_type'] in ["classification",
                               "regression",
                               "mil_classification"]:
        from miacag.dataloader.Classification.get_dataloader_classification import \
            ClassificationLoader
        CL = ClassificationLoader(config)
        train_loader, val_loader, train_ds, val_ds = \
            CL.get_classification_loader_train(config)
        val_loader.sampler.data_source.data = \
            val_loader.sampler.data_source.data * \
            config['loaders']['val_method']['samples']
    elif config['task_type'] == 'segmentation':
        from dataloader.Segmentation.get_dataloader_segmentation import \
            SegmentationLoader
        SL = SegmentationLoader()
        train_loader, val_loader = SL.get_segmentation_loader_train(config)

        val_loader.sampler.data_source.data = \
            val_loader.sampler.data_source.data * \
            config['loaders']['val_method']['samples']
    elif config['task_type'] == "representation_learning":
        from dataloader.Representation.get_dataloader_representation import \
            RepresentationLoader
        RL = RepresentationLoader()
        train_loader, val_loader = RL.get_representation_loader_train(config)
        #increase size of validation loaders
        for val_l_idx in range(0, len(val_loader)):
            val_loader[val_l_idx].sampler.data_source.data = \
                val_loader[val_l_idx].sampler.data_source.data * \
                config['loaders']['val_method']['samples']
    else:
        raise ValueError(
                "Data type is not implemented")

    return train_loader, val_loader, train_ds, val_ds
    
#    return train_loader, val_loader, train_ds, val_ds

def get_dataloader_test(config):
    if config['task_type'] in ["classification", "regression", "mil_classification"]:
        from miacag.dataloader.Classification.get_dataloader_classification import \
            ClassificationLoader
        CL = ClassificationLoader(config)
        CL.get_classificationloader_patch_lvl_test(config)
        # CL.val_loader.sampler.data_source.data = \
        #     CL.val_loader.sampler.data_source.data * \
        #     config['loaders']['val_method']['patches']
        return CL

    elif config['task_type'] == 'image2image':
        from miacag.dataloader.get_dataloader_segmentation import \
            SegmentationLoader
        SL = SegmentationLoader()
        test_loader = SL.get_segmentationloader_test(config)
        return test_loader
    else:
        raise ValueError("Data test loader mode is not implemented %s" % repr(
                    config['task_type']))
