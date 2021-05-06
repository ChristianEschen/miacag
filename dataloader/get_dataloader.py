import torch


def get_data_from_loader(data, config, device, val_phase=False):
    if config['task_type'] in ['image2scalar']:
        inputs = data['inputs'].to(device)
        labels = data['labels'].long().to(device)
    elif config['task_type'] == 'image2image':
        inputs = data['inputs'].to(device)
        labels = data['seg'].to(device)
    elif config['task_type'] == "representation_learning":
        if val_phase is False:
            if config['loaders']['backend'] != 'torchvision':
                inputs = data['inputs'].to(device)
                if config['model_dimension'] == '2D':
                    inputs = torch.cat(
                        (torch.unsqueeze(inputs[::2, :, :, :], dim=1),
                            torch.unsqueeze(inputs[1::2, :, :, :], dim=1)),
                        dim=1)
                elif config['model_dimension'] == '3D':
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
            if config['loaders']['backend'] != 'torchvision':
                inputs = data['inputs'].to(device)
                labels = data['labels'].long().to(device)
            else:
                inputs = data[0].to(device)
                labels = data[1].long().to(device)
    else:
        raise ValueError(
                "Data type is not implemented")
    return inputs, labels


def get_dataloader_train(config):
    if config['task_type'] in ['classification']:
        from dataloader.get_dataloader_classification import \
            ClassificationLoader
        CL = ClassificationLoader()
        train_loader, val_loader = CL.get_classification_loader_train(config)
        return train_loader, val_loader
    elif config['task_type'] == 'image2image':
        from dataloader.get_dataloader_segmentation import \
            SegmentationLoader
        SL = SegmentationLoader()
        train_loader, val_loader = SL.get_segmentation_loader_train(config)
        return train_loader, val_loader
    elif config['task_type'] == "representation_learning":
        from dataloader.Representation.get_dataloader_representation import \
            RepresentationLoader
        RL = RepresentationLoader()
        train_loader, val_loader = RL.get_representation_loader_train(config)
        return train_loader, val_loader
    else:
        raise ValueError(
                "Data type is not implemented")


def get_dataloader_test(config):
    if config['task_type'] in ['image2scalar']:
        from dataloader.get_dataloader_classification import \
            ClassificationLoader
        CL = ClassificationLoader()
        test_loader = CL.get_test_type(config)
        return test_loader
    elif config['task_type'] == 'image2image':
        from dataloader.get_dataloader_segmentation import \
            SegmentationLoader
        SL = SegmentationLoader()
        test_loader = SL.get_segmentationloader_test(config)
        return test_loader
    else:
        raise ValueError("Data test loader mode is not implemented %s" % repr(
                    config['task_type']))
