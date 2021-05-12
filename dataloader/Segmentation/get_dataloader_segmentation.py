from torch.utils.data import DataLoader
import torch
from monai.data import list_data_collate


class SegmentationLoader():
    def get_segmentation_loader_train(self, config):
        if config['model_name'] in ['UNet3D', 'DYNUNet3D']:
            from dataloader.dataloader_monai_segmentation_3D import \
                train_monai_segmentation_loader, \
                val_monai_segmentation_loader
            if config['loaders']['task'] == 'costum':
                train_loader = train_monai_segmentation_loader(
                    config['loaders']['TraindataRoot'],
                    config['loaders']['TraindataCSV'],
                    config)
                val_loader = val_monai_segmentation_loader(
                    config['loaders']['ValdataRoot'],
                    config['loaders']['ValdataCSV'],
                    config)

            elif config['loaders']['task'] != 'costum':
                raise ValueError("Task is not implemented")
        elif config['model_name'] == 'UNet2D':

            if config['loaders']['format'] == 'nifty':
                from dataloader.dataloader_monai_segmentation_3D import \
                    train_monai_segmentation_loader, \
                    val_monai_segmentation_loader
                train_loader = train_monai_segmentation_loader(
                        config['loaders']['TraindataRoot'],
                        config['loaders']['TraindataCSV'],
                        config)
                val_loader = val_monai_segmentation_loader(
                    config['loaders']['ValdataRoot'],
                    config['loaders']['ValdataCSV'],
                    config)

            elif config['loaders']['format'] == 'rgb':
                from dataloader.dataloader_monai_2D_RGB import \
                    train_monai_segmentation_loader, \
                    val_monai_segmentation_loader, \
                    val_monai_loader_sliding_window
                train_loader = train_monai_segmentation_loader(
                        config['loaders']['TraindataRoot'],
                        config['loaders']['TraindataCSV'],
                        config)
                if config['loaders']['val_method']['type'] == 'sliding_window':
                    val_loader = val_monai_loader_sliding_window(
                        config['loaders']['ValdataRoot'],
                        config['loaders']['ValdataCSV'],
                        config)

                elif config['loaders']['val_method']['type'] == 'patches':
                    val_loader = val_monai_segmentation_loader(
                        config['loaders']['ValdataRoot'],
                        config['loaders']['ValdataCSV'],
                        config)
            else:
                raise ValueError("Task is not implemented")
        else:
            raise ValueError("Unknown model")
        train_loader = DataLoader(train_loader(),
                                    batch_size=config['loaders']['batchSize'],
                                    shuffle=True,
                                    num_workers=config['loaders']['numWorkers'],
                                    collate_fn=list_data_collate,
                                    pin_memory=torch.cuda.is_available(),)
        with torch.no_grad():
            val_loader = DataLoader(
                val_loader(),
                batch_size=1 if
                config['loaders']['val_method']['type'] == 'sliding_window'
                else config['loaders']['batchSize'],
                shuffle=False,
                num_workers=config['loaders']['numWorkers'],
                collate_fn=list_data_collate,
                pin_memory=torch.cuda.is_available(),)
        return train_loader, val_loader

    def get_segmentationloader_test(self, config):
        if config['loaders']['format'] == 'nifty':
            from dataloader.dataloader_monai_segmentation_3D import \
                val_monai_segmentation_loader
            test_loader = val_monai_segmentation_loader(
                    config['loaders']['TestdataRoot'],
                    config['loaders']['TestdataCSV'],
                    config)
            test_loader = DataLoader(test_loader(),
                                     batch_size=config['loaders']['batchSize'],
                                     shuffle=True,
                                     num_workers=config['loaders']['numWorkers'],
                                     collate_fn=list_data_collate,
                                     pin_memory=torch.cuda.is_available(),)
        elif config['loaders']['format'] == 'rgb':
            from dataloader.dataloader_monai_2D_RGB import \
                val_monai_loader_sliding_window
            test_loader = val_monai_loader_sliding_window(
                    config['loaders']['TestdataRoot'],
                    config['loaders']['TestdataCSV'],
                    config)
            test_loader = DataLoader(test_loader(),
                                     batch_size=config['loaders']['batchSize'],
                                     shuffle=True,
                                     num_workers=config['loaders']['numWorkers'],
                                     collate_fn=list_data_collate,
                                     pin_memory=torch.cuda.is_available(),)
        else:
            raise ValueError("Task format is not implemented")
        return test_loader
