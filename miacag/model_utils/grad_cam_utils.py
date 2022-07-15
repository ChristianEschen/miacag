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
    return img[:, :, :, startz:startz+cropz]

def prepare_cv2_img(img, label, mask, data_path,
                    path_name,
                    patientID,
                    studyInstanceUID,
                    seriesInstanceUID,
                    SOPInstanceUID,
                    config):

    path = os.path.join(
        config['output_directory'],
        'saliency',
        path_name,
        patientID,
        studyInstanceUID,
        seriesInstanceUID,
        SOPInstanceUID,
        label)
    if not os.path.isdir(path):
        mkDir(path)
    img = img[0, 0, :, :, :]
    img = np.expand_dims(img, 2)
   # img2 = pydicom.read_file(data_path[0]).pixel_array
    img2 = sitk.ReadImage(data_path[0])
    img2 = sitk.GetArrayFromImage(img2)
    img2 = np.transpose(img2, (1, 2, 0))
    img2 = np.expand_dims(img2, 2)
    img2 = crop_center(img2, img.shape[-1])
    mask = mask[0, 0, :, :, :]
    mask = resizeVolume(mask, (img2.shape[0], img2.shape[1]))
  #  mask = np.expand_dims(mask, 2)
    for i in range(0, img2.shape[3]):
        input2d = img[:, :, :, i]
        input2d_2 = img2[:, :, :, i]
        cam, heatmap, input2d_2 = show_cam_on_image(
            input2d_2,
            mask[:, :, i]) # mask[:,:,:,i]

        # input2d = np.flip(np.rot90(np.rot90(np.rot90(input2d))), 1)
        # plt.imshow(input2d, cmap="gray", interpolation="None")
        # plt.colorbar()
        # plt.axis('off')
        # plt.savefig(os.path.join(path, 'input{}.png'.format(i)))
        # plt.clf()

        #input2d_2 = np.flip(np.rot90(np.rot90(np.rot90(input2d))), 1)
        plt.imshow(input2d_2, cmap="gray", interpolation="None")
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'input2{}.png'.format(i)))
        plt.clf()

        plt.imshow(input2d_2, cmap="gray", interpolation="None")
       # plt.colorbar()
        plt.savefig(os.path.join(path, 'input2_no_colormap{}.png'.format(i)))
        plt.clf()


        #cam = np.flip(np.rot90(np.rot90(np.rot90(cam))), 1)
        plt.imshow(cam, cmap="jet", interpolation="None")
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'cam{}.png'.format(i)))
        plt.clf()

        #heatmap = np.flip(np.rot90(np.rot90(np.rot90(heatmap))), 1)
        plt.imshow(heatmap, cmap="jet", interpolation="None")
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'heatmap{}.png'.format(i)))
        plt.clf()

        plt.imshow(heatmap, cmap="jet", interpolation="None")
        #plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'heatmap_no_colorbar{}.png'.format(i)))
        plt.clf()

        fig = plt.figure(figsize=(16, 12))
        #ax = plt.gca()
        fig.add_subplot(1, 2, 1)
        # f, axarr = plt.subplots(1, 2)
       # divider = make_axes_locatable(ax)
        plt.imshow(input2d_2, cmap="gray", interpolation="None")
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        #plt.colorbar(im, cax)
        # plt.colorbar()
        plt.axis('off')

        fig.add_subplot(1, 2, 2)
        # f, axarr = plt.subplots(1, 2)
        plt.imshow(heatmap, cmap="jet", interpolation="None")
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #plt.colorbar(im2, cax)
        #plt.colorbar()
        plt.axis('off')

        plt.savefig(os.path.join(path, 'twoplots{}.png'.format(i)))
        plt.clf()

       # axarr[0, 1].imshow(image_datas[1])

def show_cam_on_image(img, mask):
    img = normalize(img)
    mask = normalize(mask)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / (255 + 1e-6)
    cam = heatmap + np.float32(img)
    cam = cam / (np.max(cam) + 1e-6)
    return cam, heatmap, img

def calc_saliency_maps(model, inputs, config):
    if config['loaders']['use_amp'] is True:
        with torch.cuda.amp.autocast():
            saliency = SaliencyInferer(
                cam_name="GradCAM",
                target_layers='module.encoder.6')
    else:
        if config['model']['backbone'] == 'r2plus1d_18':
            layer_name = 'module.encoder.4.1.relu'
        elif config['model']['backbone'] == 'x3d_s':
            layer_name = 'module.encoder.5.post_conv'
            layer_name = 'module.encoder.4.res_blocks.6.activation'
        elif config['model']['backbone'] == 'debug_3d':
            layer_name = 'module.encoder.layer1'
        elif config['model']['backbone'] in ['MVIT-16', 'MVIT-32']:
            layer_name = "module.encoder.blocks.15.attn.pool_v"
        else:
            layer_name = 'module.encoder.5.post_conv'
        saliency = SaliencyInferer(
                cam_name="GradCAM",
                target_layers=layer_name)

        cams = []
        for c, label in enumerate(config['labels_names']):
            model_copy = prepare_model_for_sm(model, config, c)
            saliency = SaliencyInferer(
                cam_name="GradCAM",
                target_layers=layer_name)
            cam = saliency(network=model_copy.module, inputs=inputs)
            cam = cam[0:1, :, :, :, :]
            cams.append(cam)
        return cams, config['labels_names']


def prepare_model_for_sm(model, config, c):
    if config['task_type'] in ['regression', 'classification']:
        model.module.module.fc = model.module.module.fcs[c]
        bound_method = model.module.module.forward_saliency.__get__(
            model, model.module.module.__class__)
        setattr(model.module.module, 'forward', bound_method)
    elif config['task_type'] == 'mil_classification':
        copy_model = copy.deepcopy(model)
        copy_model.module.module.attention = model.module.module.attention[c]
        copy_model.module.module.fcs = model.module.module.fcs[c]
        if model.module.module.transformer is not None:
            copy_model.module.module.transformer = \
                model.module.module.transformer[c]
        bound_method = copy_model.module.module.forward_saliency.__get__(
            copy_model, copy_model.module.module.__class__)
        setattr(copy_model.module.module, 'forward', bound_method)
    return copy_model
