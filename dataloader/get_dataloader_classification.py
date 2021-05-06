from torch.utils.data import DataLoader
import torch
from monai.data import list_data_collate


class ClassificationLoader():
    def get_classification_loader_train(self, config):
        if config['loaders']['format'] == 'avi':
            from dataloader.dataloader_base_video import \
                getVideoTrainTransforms, getVideoTestTransforms
            from dataloader.dataloader_avi_video import VideoDataloaderAVITrain

            transforms_train = getVideoTrainTransforms(
                nr_frames=config['loaders']['Crop_depth'],
                crop_size=(config['loaders']['Crop_height'],
                           config['loaders']['Crop_width']))
            transforms_val = getVideoTestTransforms(
                nr_frames=config['loaders']['Crop_depth'],
                crop_size=(config['loaders']['Crop_height'],
                           config['loaders']['Crop_width']))

            train_loader = VideoDataloaderAVITrain(
                config['loaders']['TraindataRoot'],
                config['loaders']['TraindataCSV'],
                transforms_train)

            train_loader = DataLoader(
                train_loader,
                batch_size=config['loaders']['batchSize'],
                num_workers=config['loaders']['numWorkers'],
                sampler=train_loader.sampler)

            val_loader = VideoDataloaderAVITrain(
                config['loaders']['TraindataRoot'],
                config['loaders']['TraindataCSV'],
                transforms_val)

            with torch.no_grad():
                val_loader = DataLoader(
                    val_loader,
                    batch_size=config['loaders']['batchSize'],
                    num_workers=config['loaders']['numWorkers'],
                    shuffle=False)
            return train_loader, val_loader

        elif config['loaders']['format'] == 'nifty':
            from dataloader.dataloader_monai_classification_3D import \
                train_monai_classification_loader
            from dataloader.dataloader_monai_classification_3D import \
                val_monai_classification_loader
            train_loader = train_monai_classification_loader(
                config['loaders']['TraindataRoot'],
                config['loaders']['TraindataCSV'],
                config)

            if config['loaders']['val_method']['type'] == 'patches':
                val_loader = val_monai_classification_loader(
                    config['loaders']['ValdataRoot'],
                    config['loaders']['ValdataCSV'],
                    config)
            else:
                raise ValueError("Invalid validation moode %s" % repr(
                    config['loaders']['val_method']['type']))
            train_loader = DataLoader(
                train_loader(),
                batch_size=config['loaders']['batchSize'],
                shuffle=True,
                num_workers=config['loaders']['numWorkers'],
                collate_fn=list_data_collate,
                pin_memory=torch.cuda.is_available(),)
            with torch.no_grad():
                val_loader = DataLoader(
                    val_loader(),
                    batch_size=config['loaders']['batchSize'],
                    shuffle=False,
                    num_workers=config['loaders']['numWorkers'],
                    collate_fn=list_data_collate,
                    pin_memory=torch.cuda.is_available(),)
            return train_loader, val_loader
        elif config['loaders']['format'] == 'rgb':
            from dataloader.Representation._2D.\
                    dataloader_torchvision_representation_2D_RGB \
                    import train_torchvision_representation_loader \
                    as train_loader_rep
            from dataloader.Representation._2D.\
                    dataloader_torchvision_representation_2D_RGB \
                    import val_torchvision_representation_loader \
                    as val_loader_rep
            train_loader = train_loader_rep(
                config['loaders']['TraindataRoot'],
                config['loaders']['TraindataCSV'],
                config,
                use_complete_data=False)

            val_phase_train_loader = train_loader_rep(
                config['loaders']['ValdataRoot'],
                config['loaders']['ValdataCSV'],
                config,
                use_complete_data=False)
            val_phase_train_loader_metric = val_loader_rep(
                config['loaders']['TraindataRoot'],
                config['loaders']['TraindataCSV'],
                config,
                use_complete_data=True)
            val_phase_val_loader_metric = val_loader_rep(
                config['loaders']['ValdataRoot'],
                config['loaders']['ValdataCSV'],
                config,
                use_complete_data=True)
            if config['loaders']['store_memory'] is True:
                train_loader = datasets.CIFAR10(root=config['loaders']['TraindataRoot'],
                                                train=True,
                                                download=True,
                                                transform=train_loader().transform)
                val_loader = datasets.CIFAR10(root=config['loaders']['ValdataRoot'],
                                        train=False,
                                        download=True,
                                        transform=val_phase_train_loader().transform)
                train_loader = DataLoader(
                    train_loader,
                    drop_last=True,
                    batch_size=config['loaders']['batchSize'],
                    shuffle=True,
                    num_workers=config['loaders']['numWorkers'],
                    collate_fn=list_data_collate,
                    pin_memory=torch.cuda.is_available(),)

                val_phase_train_loader = DataLoader(
                    val_loader,
                    drop_last=True,
                    batch_size=config['loaders']['batchSize'],
                    shuffle=False,
                    num_workers=config['loaders']['numWorkers'],
                    collate_fn=list_data_collate,
                    pin_memory=torch.cuda.is_available(),)
                return train_loader, val_loader
            raise ValueError("Data type is not implemented")

    def get_test_type(self, config):
        if config['loaders']['val_method']['type'] == 'patch_lvl':
            test_loader = self.get_classificationloader_patch_lvl_test(config)
            return test_loader
        elif config['loaders']['val_method']['type'] in \
                ['image_lvl+saliency_maps', 'image_lvl']:
            test_loader = self.get_classificationloader_image_lvl_test(config)
            return test_loader
        else:
            raise ValueError(
                "Data test loader type is not implemented %s" % repr(
                    config['loaders']['type']))

    def get_classificationloader_patch_lvl_test(self, config):
        if config['loaders']['format'] == 'avi':
            from dataloader.dataloader_base_video import \
                getVideoTestTransforms
            from dataloader.dataloader_avi_video import \
                VideoDataloaderAVITrain
            transforms_test = getVideoTestTransforms(
                nr_frames=config['loaders']['Crop_depth'],
                crop_size=(config['loaders']['Crop_height'],
                           config['loaders']['Crop_width']))
            test_loader = VideoDataloaderAVITrain(
                config['loaders']['TestdataRoot'],
                config['loaders']['TestdataCSV'],
                transforms_test)
            with torch.no_grad():
                test_loader = DataLoader(test_loader,
                                         batch_size=config[
                                             'loaders']['batchSize'],
                                         num_workers=config[
                                            'loaders']['numWorkers'],
                                         shuffle=False)
            return test_loader
        elif config['loaders']['format'] == 'nifty':
            from dataloader.dataloader_monai_classification_3D \
                import test_monai_classification_loader

            test_loader = test_monai_classification_loader(
                config['loaders']['TestdataRoot'],
                config['loaders']['TestdataCSV'],
                config)
            test_loader = DataLoader(
                test_loader(),
                batch_size=config['loaders']['batchSize'],
                shuffle=False,
                num_workers=config['loaders']['numWorkers'],
                collate_fn=list_data_collate,
                pin_memory=torch.cuda.is_available(),)

            return test_loader

    def get_classificationloader_image_lvl_test(self, config):
        if config['loaders']['format'] == 'avi':
            from dataloader.dataloader_base_video import \
                getVideoTestTransforms
            from dataloader.dataloader_avi_video import \
                VideoDataloaderAVITest
            transforms_test = getVideoTestTransforms(
                nr_frames=config['loaders']['Crop_depth'],
                crop_size=(config['loaders']['Crop_height'],
                           config['loaders']['Crop_width']))
            test_loader = VideoDataloaderAVITest(
                config['loaders']['TestdataRoot'],
                config['loaders']['TestdataCSV'],
                config['loaders']['Crop_depth'],
                transforms_test)

            test_loader = DataLoader(test_loader,
                                     batch_size=1,
                                     num_workers=config[
                                         'loaders']['numWorkers'],
                                     shuffle=False)

            return test_loader
        elif config['loaders']['format'] == 'nifty':
            from dataloader.dataloader_monai_classification_3D \
                import test_monai_classification_loader

            test_loader = test_monai_classification_loader(
                config['loaders']['TestdataRoot'],
                config['loaders']['TestdataCSV'],
                config)
            test_loader = DataLoader(
                test_loader(),
                batch_size=config['loaders']['batchSize'],
                shuffle=False,
                num_workers=config['loaders']['numWorkers'],
                collate_fn=list_data_collate,
                pin_memory=torch.cuda.is_available(),)

            return test_loader
