import torch
import os


def get_data_from_loader(data, config, device, val_phase=False):
    if config['loaders']['store_memory'] is True:
        data = {
                'inputs': data[0],
                'labels': data[1]
                }
    if config['task_type'] in ['classification', 'segmentation']:
        inputs = data['inputs'].to(device)
        labels = data['labels'].long().to(device)
    elif config['task_type'] == "representation_learning":
        if val_phase is False:
            if config['loaders']['store_memory'] is False:
                inputs = data['inputs'].to(device)
                if config['model']['dimension'] == '2D':
                    inputs = torch.cat(
                        (torch.unsqueeze(inputs[::2, :, :, :], dim=1),
                            torch.unsqueeze(inputs[1::2, :, :, :], dim=1)),
                        dim=1)
                elif config['model']['dimension'] in ['3D', '2D+T']:
                    inputs = torch.cat(
                        (torch.unsqueeze(inputs[::2, :, :, :, :], dim=1),
                            torch.unsqueeze(inputs[1::2, :, :, :, :], dim=1)),
                        dim=1)
                else:
                    raise ValueError(
                            "model dimension not implemented")
            else:
                inputs = torch.cat(
                    (torch.unsqueeze(data[0][0].to(device), 1),
                     torch.unsqueeze(data[0][1].to(device), 1)),
                    dim=1)
            labels = None
        else:
            if config['loaders']['store_memory'] is False:
                inputs = data['inputs'].to(device)
                labels = data['labels'].long().to(device)
            else:
                inputs = data[0].to(device)
                labels = data[1].long().to(device)
    else:
        raise ValueError(
                "Data type is not implemented")
    if config['loaders']['mode'] == 'testing':
        return inputs, labels, data['index']
    else:
        return inputs, labels

def get_data_from_standard_Datasets(data, config, device, val_phase):
    data = {
                'inputs': data[0],
                'labels': data[1]
                }
    if config['task_type'] == "representation_learning":
        if val_phase is False:
            inputs = torch.cat(
                        (torch.unsqueeze(data[0][0].to(device), 1),
                        torch.unsqueeze(data[0][1].to(device), 1)),
                        dim=1)


def get_dataloader_train(config):
    if config['task_type'] in ['classification']:
        from dataloader.Classification.get_dataloader_classification import \
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


def get_dataloader_test(config):
    if config['task_type'] in ["classification"]:
        from dataloader.Classification.get_dataloader_classification import \
            ClassificationLoader
        CL = ClassificationLoader(config)
        CL.get_classificationloader_patch_lvl_test(config)
        # CL.val_loader.sampler.data_source.data = \
        #     CL.val_loader.sampler.data_source.data * \
        #     config['loaders']['val_method']['patches']
        return CL

    elif config['task_type'] == 'image2image':
        from dataloader.get_dataloader_segmentation import \
            SegmentationLoader
        SL = SegmentationLoader()
        test_loader = SL.get_segmentationloader_test(config)
        return test_loader
    else:
        raise ValueError("Data test loader mode is not implemented %s" % repr(
                    config['task_type']))
