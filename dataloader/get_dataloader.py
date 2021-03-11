def get_data_from_loader(data, config, device):
    if config['loaders']['task_type'] in ['image2scalar']:
        inputs = data['inputs'].to(device)
        labels = data['labels'].to(device)
    elif config['loaders']['task_type'] == 'image2image':
        inputs = data['inputs'].to(device)
        labels = data['seg'].to(device)
    else:
        raise ValueError(
                "Data type is not implemented")
    return inputs, labels


def get_dataloader_train(config):
    if config['loaders']['task_type'] in ['image2scalar']:
        from dataloader.get_dataloader_classification import \
            ClassificationLoader
        CL = ClassificationLoader()
        train_loader, val_loader = CL.get_classification_loader_train(config)
        return train_loader, val_loader
    elif config['loaders']['task_type'] == 'image2image':
        from dataloader.get_dataloader_segmentation import \
            SegmentationLoader
        SL = SegmentationLoader()
        train_loader, val_loader = SL.get_segmentation_loader_train(config)
        return train_loader, val_loader
    else:
        raise ValueError(
                "Data type is not implemented")


def get_dataloader_test(config):
    if config['loaders']['task_type'] in ['image2scalar']:
        from dataloader.get_dataloader_classification import \
            ClassificationLoader
        CL = ClassificationLoader()
        test_loader = CL.get_test_type(config)
        return test_loader
    elif config['loaders']['task_type'] == 'image2image':
        from dataloader.get_dataloader_segmentation import \
            SegmentationLoader
        SL = SegmentationLoader()
        test_loader = SL.get_segmentationloader_test(config)
        return test_loader
    else:
        raise ValueError("Data test loader mode is not implemented %s" % repr(
                    config['loaders']['task_type']))
