from model_utils.eval_utils import val_one_epoch
from model_utils.eval_utils import eval_one_step
import os
import json


class TestPipeline():
    def get_test_pipeline(self, model, criterion, config, test_loader,
                          device, init_metrics, increment_metrics,
                          normalize_metrics,
                          running_metric_test, running_loss_test):

        if config['loaders']['task_type'] == "image2scalar":
            self.get_test_classification_pipeline(model, criterion,
                                                  config, test_loader,
                                                  device, init_metrics,
                                                  increment_metrics,
                                                  normalize_metrics,
                                                  running_metric_test,
                                                  running_loss_test)
        elif config['loaders']['task_type'] == "image2image":
            self.get_test_segmentation_pipeline(model, criterion,
                                                config, test_loader,
                                                device, init_metrics,
                                                increment_metrics,
                                                normalize_metrics,
                                                running_metric_test,
                                                running_loss_test)
        else:
            raise ValueError("Task type is not implemented")

    def get_test_classification_pipeline(self, model, criterion,
                                         config, test_loader,
                                         device, init_metrics,
                                         increment_metrics,
                                         normalize_metrics,
                                         running_metric_test,
                                         running_loss_test):

        if config['loaders']['val_method']['type'] == 'patch_lvl':
            metrics = val_one_epoch(model, criterion, config,
                                    test_loader, device,
                                    running_metric_val=running_metric_test,
                                    running_loss_val=running_loss_test,
                                    saliency_maps=False)

        elif config['loaders']['val_method']['type'] in \
                ['image_lvl', 'image_lvl+saliency_maps']:
            if config['loaders']['backend'] == 'monai':
                from models.image2scalar_utils.utils_3D.test_utils_monai \
                    import _test_one_epoch
                if config['loaders']['val_method']['type'] == \
                        'image_lvl+saliency_maps':
                    saliency_maps = True
                else:
                    saliency_maps = False
                metrics = _test_one_epoch(
                    model, criterion, config,
                    test_loader, device,
                    eval_one_step,
                    init_metrics,
                    increment_metrics,
                    normalize_metrics,
                    saliency_maps,
                    running_metric_test=running_metric_test,
                    running_loss_test=running_loss_test)
            elif config['loaders']['backend'] == 'costum':
                from models.image2scalar_utils.utils_3D.test_utils import \
                    _test_one_epoch
                if config['loaders']['val_method']['type'] == \
                        'image_lvl+saliency_maps':
                    saliency_maps = True
                else:
                    saliency_maps = False
                metrics = _test_one_epoch(
                    model, criterion, config,
                    test_loader, device,
                    eval_one_step,
                    init_metrics,
                    increment_metrics,
                    normalize_metrics,
                    saliency_maps,
                    running_metric_test=running_metric_test,
                    running_loss_test=running_loss_test)
            else:
                raise ValueError(
                    "backend methodd not implemented %s" % repr(
                        config['loaders']['backend']))
        else:
            raise ValueError(
                "test pipeline is not implemented %s" % repr(
                    config['loaders']['val_method']['type']))
        with open(os.path.join(config['model_path'],
                               'test_log.txt'), 'w') as file:
            file.write(json.dumps({**metrics, **config},
                                  sort_keys=True, indent=4,
                                  separators=(',', ': ')))

    def get_test_segmentation_pipeline(self, model, criterion,
                                       config, test_loader,
                                       device, init_metrics, increment_metrics,
                                       normalize_metrics,
                                       running_metric_test, running_loss_test):
        from models.image2image_utils.utils_3D.test_utils_img2img_monai \
                import slidingWindowTest
        testModule = slidingWindowTest(model, criterion, config,
                                       test_loader, device,
                                       running_metric_val=running_metric_test,
                                       running_loss_val=running_loss_test,
                                       saliency_maps=False)
        testModule()
