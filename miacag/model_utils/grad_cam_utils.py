import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from miacag.metrics.metrics_utils import mkDir
import pydicom
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
import SimpleITK as sitk
from monai.inferers import SimpleInferer, SaliencyInferer
import copy
from miacag.models.get_encoder import Identity
from miacag.model_utils.GradCam_model import GradCAM, ClassifierOutputTarget
#from miacag.model_utils.GradCam_monai import GradCAM
import torch.optim
from miacag.models.BuildModel import ModelBuilder
from monai.transforms import LoadImage


def resize3dVolume(img, output_size):
    factors = (output_size[0]/img.shape[0],
               output_size[1]/img.shape[1],
               output_size[2]/img.shape[2])

    new_array = zoom(img, (factors[0], factors[1], factors[2]))
    return new_array

def resizeVolume(img, output_size):
    factors = (output_size[0]/img.shape[0],
               output_size[1]/img.shape[1])

    new_array = zoom(img, (factors[0], factors[1], 1))
    return new_array


def normalize(img):
    img = (img - np.min(img)) / \
        (np.amax(img)-np.amin(img)) # + 1e-8
    return img


def crop_center(img, cropz):
    z = img.shape[-1]
    startz = z//2-(cropz//2)
    return img[:, :, startz:startz+cropz]


def prepare_img_and_mask(data_path, img, mask, config):
    if config['model']['dimension'] in ['2D+T', '3D']:
        loaded_img, mask = prepare_img_and_mask_3d(data_path, mask)
    elif config['model']['dimension'] in ['2D']:
        loaded_img, mask = prepare_img_and_mask_2d(img, mask)
    else:
        raise ValueError('not implemented')
    return loaded_img, mask


def prepare_img_and_mask_2d(img, mask):
    img_loaded = np.expand_dims(img[0,0,:,:], -1)
    mask = mask[0, 0, :, :]
    return img_loaded, mask


def prepare_img_and_mask_3d(data_path, mask):
    img_loaded, meta = LoadImage()(data_path[0])
    img_loaded = crop_center(img_loaded, mask.shape[-1])
    img_loaded = np.expand_dims(img_loaded, 2)

    mask = mask[0, 0, :, :, :]
    mask = resizeVolume(mask, (img_loaded.shape[0], img_loaded.shape[1]))
    return img_loaded, mask

    
def prepare_cv2_img(img, label, mask, data_path,
                    path_name,
                    patientID,
                    studyInstanceUID,
                    seriesInstanceUID,
                    SOPInstanceUID,
                    config,
                    patch=0):

    path = os.path.join(
        config['model']['pretrain_model'],
        'saliency',
        path_name,
        patientID,
        studyInstanceUID,
        seriesInstanceUID,
        SOPInstanceUID,
        label)
    if not os.path.isdir(path):
        mkDir(path)

    img_loaded, mask = prepare_img_and_mask(data_path, img, mask, config)
    if config['model']['dimension'] in ['2D+T', '3D']:
        plot_3d(img_loaded, mask, path)

    elif config['model']['dimension'] in ['2D']:
        plot_2d(img_loaded, mask, path, patch)
    else:
        raise ValueError('not implemented')


def plot_3d(img_loaded, mask, path):
    for i in range(0, img_loaded.shape[-1]):
        input2d_2 = img_loaded[:, :, :, i]
        cam, heatmap, input2d_2 = show_cam_on_image(
            input2d_2,
            mask[:, :, i])
        plot_gradcam(input2d_2, path, cam, heatmap, i)


def plot_2d(input2d_2, mask, path, i):
   # input2d_2 = img_loaded[:, :, :, i]
    cam, heatmap, input2d_2 = show_cam_on_image(
        input2d_2,
        mask)
    plot_gradcam(input2d_2, path, cam, heatmap, i)


def plot_gradcam(input2d_2, path, cam, heatmap, i):
    input2d_2 = np.flip(np.rot90(np.rot90(np.rot90(input2d_2))), 1)
    plt.imshow(input2d_2, cmap="gray", interpolation="None")
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(path, 'input2{}.png'.format(i)))
    plt.clf()
    plt.close()

    plt.imshow(input2d_2, cmap="gray", interpolation="None")
    plt.savefig(os.path.join(path, 'input2_no_colormap{}.png'.format(i)))
    plt.clf()
    plt.close()

    cam = np.flip(np.rot90(np.rot90(np.rot90(cam))), 1)
    plt.imshow(cam, cmap="jet", interpolation="None")
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(path, 'cam{}.png'.format(i)))
    plt.clf()
    plt.close()

    heatmap = np.flip(np.rot90(np.rot90(np.rot90(heatmap))), 1)
    plt.imshow(heatmap, cmap="jet", interpolation="None")
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(path, 'heatmap{}.png'.format(i)))
    plt.clf()
    plt.close()

    plt.imshow(heatmap, cmap="jet", interpolation="None")
    plt.axis('off')
    plt.savefig(os.path.join(path, 'heatmap_no_colorbar{}.png'.format(i)))
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(16, 12))
    fig.add_subplot(1, 2, 1)
    plt.imshow(input2d_2, cmap="gray", interpolation="None")

    plt.axis('off')

    fig.add_subplot(1, 2, 2)
    plt.imshow(heatmap, cmap="jet", interpolation="None")
    plt.axis('off')

    plt.savefig(os.path.join(path, 'twoplots{}.png'.format(i)))
    plt.clf()
    plt.close()



def show_cam_on_image(img, mask):
    img = normalize(img)
    mask = normalize(mask)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / (255 + 1e-6)
    cam = heatmap + np.float32(img)
    cam = cam / (np.max(cam) + 1e-6)
    return cam, heatmap, img


def calc_saliency_maps(model, inputs, config, device, c):
    if config['model']['backbone'] == 'r2plus1_18':
        layer_name = 'module.encoder.4.1.relu'
    elif config['model']['backbone'] == 'x3d_s':
        layer_name = 'module.encoder.5.post_conv'
        layer_name = 'module.encoder.4.res_blocks.6.activation'
    elif config['model']['backbone'] == 'debug_3d':
        layer_name = 'module.encoder.layer1'
    elif config['model']['backbone'] in ["mvit_base_16x4", 'mvit_base_32x3']:
        layer_name = "module.module.encoder.blocks.15.attn"
        layer_name = "module.module.encoder.blocks.15.norm1"
        target_layers = [model.module.module.encoder.blocks[-1].norm1]
        #target_layers = [model.module.module.encoder.blocks[-1].mlp]
    elif config['model']['backbone'] == 'r50':
        layer_name = "module.encoder.7.2.conv3"
    else:
        layer_name = 'module.encoder.5.post_conv'

    # BuildModel = ModelBuilder(config, device)
    # model = BuildModel()
    # if config['use_DDP'] == 'True':
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[device] if config["cpu"] == "False" else None)
    model = prepare_model_for_sm(model, config, c)
    if config['model']['backbone'] in [
            "mvit_base_16x4", "mvit_base_32x3"]:
        target_layers = [model.module.module.encoder.blocks[-1].norm1]
        cam_f = GradCAM(model=model,
                        target_layers=target_layers,
                        reshape_transform=reshape_transform)
       # targets = [ClassifierOutputTarget(0)]
        cam = cam_f(input_tensor=inputs)
        cam = resize3dVolume(
            cam[0, :, :, :],
            (config['loaders']['Crop_height'],
                config['loaders']['Crop_width'],
                config['loaders']['Crop_depth']))
        cam = np.expand_dims(np.expand_dims(cam, 0), 0)
    else:
        # model.module.module.encoder[-1][-1].conv2[0][-1]
        #layer_name = "module.encoder.7.2.conv3"
        saliency = SaliencyInferer(
            cam_name="GradCAM",
            target_layers=layer_name,
            class_idx=c)
        cam = saliency(network=model.module, inputs=inputs)

    if config['model']['dimension'] in ['2D+T', "3D"]:
        return cam[0:1, :, :, :, :]
    elif config['model']['dimension'] in ['2D']:
        return cam[0:1, :, :, :]
    else:
        raise ValueError('not implemented')


def reshape_transform(tensor, height=14, width=14):
    tensor = tensor[:, 1:, :]
    tensor = tensor.unsqueeze(dim=0)
    result = torch.nn.functional.interpolate(
        tensor,
        scale_factor=(392/tensor.size(2), 1))
    result = result.reshape(result.size(0), 8, 7, 7, result.size(-1))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(0, 4, 1, 2, 3)
    return result


def prepare_model_for_sm(model, config, c):
    if config['task_type'] in ['regression', 'classification']:
       # copy_model = copy.deepcopy(model)
       # copy_model.module.module.fc = copy_model.module.module.fcs[c]
        bound_method = model.module.module.forward_saliency.__get__(
            model, model.module.module.__class__)
        setattr(model.module.module, 'forward', bound_method)
        return model
    elif config['task_type'] == 'mil_classification':
        c = 0 
       # print('only implemented for one head!')
       # model = model
       # model.module.module.attention = model.module.module.attention[c]
       # model.module.module.fcs = model.module.module.fcs[c]
        #if model.module.module.transformer is not None:
        #    model.module.module.transformer = \
        #        model.module.module.transformer[c]
        bound_method = model.module.module.forward_saliency.__get__(
            model, model.module.module.__class__)
        setattr(model.module.module, 'forward', bound_method)
        return model
