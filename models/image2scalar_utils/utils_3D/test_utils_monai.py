import monai
from models.image2scalar_utils.utils_3D.gradcam import GuidedBackpropReLUModel


def _test_one_epoch(model, criterion, config,
                    video_loader_test, device, one_step_eval,
                    init_metrics,
                    increment_metrics,
                    normalize_metrics,
                    saliency_maps,
                    running_metric_test=0.0, running_loss_test=0.0):
    print('NOT IMPLEMENTED')
    gradcam = monai.visualize.GradCAM(
        nn_module=model, target_layers='module.model.layer4.2.relu')
    for i_batch, sample_batched in enumerate(video_loader_test):

        print('i_batch', i_batch)
        patches = sample_batched['inputs'].shape[1]
        images = []
        cam = []
        heat = []
        guide_backprop = []
        running_loss_patch, _ = init_metrics(config['loss']['name'],
                                             config['model']['classes'])
        running_metric_patch, _ = init_metrics(config['eval_metric']['name'],
                                               config['model']['classes'])
        for i in range(0, patches):
            patch_vid = sample_batched['inputs'][:, i, :, :, :]
            patch_vid = patch_vid.to(device)

            sample_batched['labels'] = sample_batched['labels'].to(device)
            output_patch, loss_patch, metrics_patch = \
                one_step_eval(model, patch_vid,
                              sample_batched['labels'],
                              device, criterion,
                              config,
                              saliency_maps=saliency_maps)
            res_cam = gradcam(x=patch_vid,
                              class_idx=sample_batched['labels'])
            res_guided_backP = GuidedBackpropReLUModel(model=model,
                                                       use_cuda=device)
            print('js')