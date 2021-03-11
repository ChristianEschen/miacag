import torch
from model_utils.get_model import get_model
import torch.nn as nn
import os
from models.image2scalar_utils.utils_3D.gradcam import GuidedBackpropReLUModel
from models.image2scalar_utils.utils_3D.gradcam import GradCam
from models.image2scalar_utils.utils_3D.gradcam import show_cam_on_image
from models.image2scalar_utils.utils_3D.gradcam import normalize
import numpy as np
import imageio


def build_model(config, device):
    model = get_model(config)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                  'model.pt')))
    model.to(device)
    return model


def reorder_image(image):
    image = np.squeeze(image)
    image = np.rollaxis(np.rollaxis(image, 2), 2)
    return image


def guide_bp(model, config, inputs, device, labels):
    model = build_model(config, device)
    guide = GuidedBackpropReLUModel(model=model, use_cuda=device)
    inputs = inputs.clone()
    gb = np.zeros((inputs.shape[2],
                   inputs.shape[3], inputs.shape[4],
                   inputs.shape[1]), dtype=np.float32)
    inputs = inputs.requires_grad_(True)
    guidebp = guide(inputs, index=labels)
    for i in range(inputs.shape[2]):
        gp_img = guidebp[:, i, :, :]
        gb[i, :, :, :] = reorder_image(gp_img)[:, :, ::-1]
    return gb


def produce_saliency(inputs, labels, model, config, device):
    grad_cam = GradCam(model=model.module.model,
                       feature_module=model.module.model.layer4,
                       target_layer_names=["2"], use_cuda=device)
    mask = grad_cam(inputs, labels)

    gb = guide_bp(model, config, inputs, device, labels)
    gb = gb * np.repeat(mask[:, :, :, np.newaxis], 3, axis=3)
    cam = np.zeros((inputs.shape[2],
                    inputs.shape[3], inputs.shape[4],
                    inputs.shape[1]), dtype=np.float32)
    heatmap = np.zeros((inputs.shape[2],
                        inputs.shape[3], inputs.shape[4],
                        inputs.shape[1]), dtype=np.float32)

    for i in range(inputs.shape[2]):
        image = inputs[:, :, i, :, :].cpu().numpy()
        image = reorder_image(image)
        c, h = show_cam_on_image(image,
                                 mask[i, :, :], i)
        cam[i, :, :, :] = c[:, :, ::-1]
        heatmap[i, :, :, :] = h[:, :, ::-1]
    return cam, heatmap, mask, gb


def mkFoldersFrame(sample_path):
    path_img = os.path.join(sample_path, 'input_image')
    path_cam = os.path.join(sample_path, 'gradCam')
    path_heat = os.path.join(sample_path, 'heatMap')
    path_gb = os.path.join(sample_path, 'guide_backprop')
    mkDir(path_img)
    mkDir(path_cam)
    mkDir(path_heat)
    mkDir(path_gb)
    return path_img, path_cam, path_heat, path_gb


def save_gradcram_frames(image, c, h, gb, frame_path, i):
    save_single_frame(image, os.path.join(frame_path[0],
                                          'input_'+str(i))+'.png')
    save_single_frame(c, os.path.join(frame_path[1],
                                      'cam_'+str(i))+'.png')
    save_single_frame(h, os.path.join(frame_path[2],
                                      'heat_'+str(i))+'.png')
    save_single_frame(gb, os.path.join(frame_path[3],
                                       'gb_'+str(i))+'.png')
    return None


def save_single_frame(image, path):
    imageio.imwrite(path, image)
    return None


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def majority_vote(metrics, patches):
    for metric in metrics:
        metrics[metric] = np.ceil(metrics[metric]/patches)
    return metrics


def _test_one_epoch(model, criterion, config,
                    video_loader_test, device, one_step_eval,
                    init_metrics,
                    increment_metrics,
                    normalize_metrics,
                    saliency_maps,
                    running_metric_test=0.0, running_loss_test=0.0):
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

            running_metric_patch = increment_metrics(running_metric_patch,
                                                     metrics_patch)
            running_loss_patch = increment_metrics(running_loss_patch,
                                                   loss_patch)

            if saliency_maps:
                if i_batch <= config['loaders']['val_method']['nr_saliency']:

                    c, h, mask, gb = produce_saliency(patch_vid,
                                                      sample_batched['labels'],
                                                      model,
                                                      config,
                                                      device)
                    patch_vid = patch_vid.cpu().detach().numpy()
                    patch_vid = np.rollaxis(np.rollaxis(np.rollaxis(
                            np.squeeze(patch_vid), 3), 3), 3)
                    patch_vid = np.uint8(255 * normalize(patch_vid))
                    c = np.uint8(255 * normalize(c))
                    h = np.uint8(255 * normalize(h))
                    gb = np.uint8(255 * normalize(gb))

                    images.append(patch_vid)
                    cam.append(c)
                    heat.append(h)
                    guide_backprop.append(gb)
        images = np.concatenate(images, axis=0)
        images = images[0:sample_batched['shape'], :, :, :]
        cam = np.concatenate(cam, axis=0)
        cam = cam[0:sample_batched['shape'], :, :, :]
        heat = np.concatenate(heat, axis=0)
        cam = cam[0:sample_batched['shape'], :, :, :]
        guide_backprop = np.concatenate(guide_backprop, axis=0)
        guide_backprop = guide_backprop[0:sample_batched['shape'], :, :, :]

        if saliency_maps:
            saliency_maps_path = os.path.join(config['model_path'],
                                              'guideGradCam')
            mkDir(saliency_maps_path)
            sample_path = os.path.join(
                         saliency_maps_path,
                         os.path.basename(
                             sample_batched['file'][0][:-4] if
                             config['loaders']['format'] == 'avi'
                             else sample_batched['file'][0][:-12]))
            mkDir(sample_path)
            imageio.mimsave(os.path.join(
                                        sample_path, 'input.gif'),
                            images)
            imageio.mimsave(os.path.join(
                                        sample_path,
                                        'gradCam.gif'),
                            cam)
            imageio.mimsave(os.path.join(
                                        sample_path, 'heatMap.gif'),
                            heat)
            imageio.mimsave(os.path.join(
                                        sample_path,
                                        'guide_backprop.gif'),
                            guide_backprop)
            if i_batch <= config['loaders']['val_method']['nr_saliency']:
                for i in range(0, int(sample_batched['shape'])):
                    frame_paths = mkFoldersFrame(sample_path)
                    save_gradcram_frames(images[i, :, :, :],
                                         cam[i, :, :, :],
                                         heat[i, :, :, :],
                                         guide_backprop[i, :, :, :],
                                         frame_paths, i)

        # majority vote for all patches.
        running_metric_patch = majority_vote(running_metric_patch, patches)

        # update mean for a sample
        running_metric_test = increment_metrics(running_metric_test,
                                                running_metric_patch)
        running_loss_test = increment_metrics(running_loss_test,
                                              running_loss_patch)

    running_metric_test = normalize_metrics(running_metric_test,
                                            len(video_loader_test))
    running_loss_test = normalize_metrics(running_loss_test,
                                          len(video_loader_test))

    for i in running_metric_test:
        print(i, running_metric_test[i])
    print('loss', running_loss_test)
    running_metric_test.update(running_loss_test)
    return running_metric_test
