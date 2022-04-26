from torch.utils.data import DataLoader
import torch
from monai.data import list_data_collate


class SegmentationLoader():
    def get_segmentation_loader_train(self, config):
        if config['model']['dimension'] in ['3D', '2D+T']:
            from dataloader.Segmentation._3D.dataloader_monai_segmentation_3D \
                import \
                train_monai_segmentation_loader, \
                val_monai_segmentation_loader
            train_loader = train_monai_segmentation_loader(
                config['TraindataRoot'],
                config['TraindataCSV'],
                config)
            val_loader = val_monai_segmentation_loader(
                config['ValdataRoot'],
                config['ValdataCSV'],
                config)
        elif config['model']['dimension'] in ['2D']:

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
                from dataloader.Segmentation._2D.dataloader_monai_segmentation_2D_RGB import \
                    train_monai_segmentation_loader, \
                    val_monai_segmentation_loader, \
                    val_monai_loader_sliding_window
                train_loader = train_monai_segmentation_loader(
                        config['TraindataRoot'],
                        config['TraindataCSV'],
                        config)
                if config['loaders']['val_method']['type'] == 'sliding_window':
                    val_loader = val_monai_loader_sliding_window(
                        config['ValdataRoot'],
                        config['ValdataCSV'],
                        config)

                elif config['loaders']['val_method']['type'] == 'patches':
                    val_loader = val_monai_segmentation_loader(
                        config['ValdataRoot'],
                        config['ValdataCSV'],
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
