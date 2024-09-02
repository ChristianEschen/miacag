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
from miacag.utils.GradCam import GradCAMR2D
from monai.visualize.visualizer import default_upsampler
import torch.nn.functional as F
import random
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


def crop_center(img, mask):
    crop_mask_d = mask.shape[-1]
    crop_img_d = img.shape[-1]
    print('mask shape', mask.shape)
    print('image shape', img.shape)
    if mask.shape[-1] == 19:
        print('stop here mask ahs', mask.shape)
    if crop_img_d <= crop_mask_d:
        # mask is biggest
        #if mask.shape[-1] > crop_img_d:
     #   mask = mask[:, :, :crop_img_d]
        start_idx =  crop_mask_d // 2 -crop_img_d//2
        
        mask = mask[:, :,:, :,start_idx:start_idx +crop_img_d]
        return img, mask
    else:
        padded_mask = np.zeros((1, 1, mask.shape[2], mask.shape[3], crop_img_d))
        startcrop_img_d = crop_img_d//2-(crop_mask_d//2)
        start_d = crop_img_d//2 - crop_mask_d // 2
        end_d = start_d + crop_mask_d
        padded_mask[:, :, :, :, start_d:end_d] = mask

        return img, padded_mask


def prepare_img_and_mask(data_path, img, mask, config):
    if config['model']['dimension'] in ['2D+T', '3D']:
        #mg_crop, mask, img_loaded, mask_img_loaded = prepare_img_and_mask_3d(data_path, mask)
        img_crop, mask, img_loaded, mask_img_loaded  = prepare_img_and_mask_3d(data_path, mask)
    elif config['model']['dimension'] in ['2D']:
        loaded_img, mask = prepare_img_and_mask_2d(img, mask)
    else:
        raise ValueError('not implemented')
    

    return img_crop, mask, img_loaded, mask_img_loaded

# def compute_np_array_with_center_gradcam(img_shape, mask):

#     mask = np.expand_dims(mask, 0)
#     mask = np.rollaxis(mask, 3,1)
#     # Extract the mask and image shape dimensions
#     _, _, mask_h, mask_w, mask_d = mask.shape
#     img_h, img_w, img_d = img_shape

#     # Initialize an array of zeros with the shape of img_shape as np array
#     padded_mask = np.zeros((1, 3, mask_h, mask_w, img_d))
#     #padded_mask = torch.zeros((1, 1, mask_h, mask_w, img_d))
    

#     # If the mask has too many slices (along the z-axis)
#     if mask_d > img_d:
#         # Calculate the starting and ending slice for cropping
#         start_d = mask_d//2 - img_d // 2
#         end_d = start_d + img_d
        
#         # Crop the mask along the z-axis
#         padded_mask = mask[:, :, :, :, start_d:end_d]

#     elif mask_d <= img_d:
#         # Calculate the starting index to place the mask in the center
#         start_d = img_d//2 - mask_d // 2
#         end_d = start_d + mask_d

#         # Resize mask to match the height and width of img_shape
#         #resized_mask = F.interpolate(mask, size=(img_h, img_w), mode='bilinear', align_corners=False)
        
#         # Insert the resized mask into the center of the padded array
#         padded_mask[:, :, :, :, start_d:end_d] = mask

#     padded_mask = padded_mask[0, :, :, :, :]
#     return padded_mask

def prepare_img_and_mask_2d(img, mask):
    img_loaded = np.expand_dims(img[0,0,:,:], -1)
    mask = mask[0, 0, :, :]
    return img_loaded, mask

def crop_center_to_mask(img, cropz):
    z = img.shape[-1]
    if z >= cropz:
        startz = z//2-(cropz//2)
        return img[:, :, startz:startz+cropz]
    else:
        padded_img = np.zeros((img.shape[0], img.shape[1], cropz))
        startcrop_img_d = cropz//2-(z//2)
        start_d = cropz//2 - z // 2
        end_d = start_d + z
        padded_img[:, :, start_d:end_d] = img
        return padded_img
        



def prepare_img_and_mask_3d(data_path, mask):
    img_loaded = LoadImage()(data_path)
    img_shape = img_loaded.shape
    mask_img_loaded = copy.deepcopy(mask)
    img_crop = copy.deepcopy(img_loaded)

    img_crop = crop_center_to_mask(img_crop, mask.shape[-1])
    img_crop = np.expand_dims(img_crop, 2)
    img_loaded, mask_img_loaded = crop_center(img_loaded, mask_img_loaded)
    img_loaded = np.expand_dims(img_loaded, 2)
    
    mask = mask[0, 0, :, :, :]
    mask = mask.cpu().numpy()

    mask = resizeVolume(mask, (img_loaded.shape[0], img_loaded.shape[1]))
    
    
    mask_img_loaded = mask_img_loaded[0, 0, :, :, :]
    mask_img_loaded = mask_img_loaded

    mask_img_loaded = resizeVolume(mask_img_loaded, (img_loaded.shape[0], img_loaded.shape[1]))
    
    # insert 
    return img_crop, mask, img_loaded, mask_img_loaded

    
def prepare_cv2_img(img, label, mask, data_path,
                    path_name,
                    patientID,
                    studyInstanceUID,
                    seriesInstanceUID,
                    SOPInstanceUID,
                    config,
                    phase_plot,
                    patch=0):

    path = os.path.join(
        phase_plot,
        'saliency',
        path_name,
        patientID,
        studyInstanceUID,
        seriesInstanceUID,
        SOPInstanceUID,
        label)
    if not os.path.isdir(path):
        mkDir(path)

    #img_loaded, mask, img_ori_shaped = prepare_img_and_mask(data_path, img, mask, config)
    img_crop, mask, img_loaded, mask_img_loaded = prepare_img_and_mask(data_path, img, mask, config)
    # save cam_3d_np as npy
  #  np.save(os.path.join(path, 'npycam3d.npy'), cam_3d_np)
    if config['model']['dimension'] in ['2D+T', '3D']:
        plot_3d(img_crop, mask, img_loaded, mask_img_loaded, path)

    elif config['model']['dimension'] in ['2D']:
        plot_2d(img_loaded, mask, path, patch)
    else:
        raise ValueError('not implemented')


def plot_3d(img_crop, mask, img_loaded, mask_img_loaded, path):
    # print('img shape original', img_shape)
    # print('img loaded final', img_loaded.shape)
    # print('mask final', mask.shape)
 #   img_crop = img_crop[0, :,]
    for i in range(0, img_crop.shape[-1]):
        input2d_2 = img_crop[:, :, :, i]
        cam, heatmap, input2d_2 = show_cam_on_image(
            input2d_2,
            mask[:, :, i])
        plot_gradcam(input2d_2, path, cam, heatmap, i)
       # cam_3d.append(cam)
    cam_3d = []
    for i in range(0, img_loaded.shape[-1]):
        input2d_2 = img_loaded[:, :, :, i]
        cam, heatmap, input2d_2 = show_cam_on_image(
            input2d_2,
            mask_img_loaded[:, :, i])
        cam_3d.append(cam)

    # stack numpy array
    cam_3d_np = np.stack(cam_3d, axis=-1)
    cam_3d_np = np.rollaxis(cam_3d_np, 2, 0)
  #  cam_3d_np = compute_np_array_with_center_gradcam(img_shape, cam_3d_np)
    np.save(os.path.join(path, 'npycam3d.npy'), cam_3d_np)


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

    # save cam as npy
   # np.save(os.path.join(path, 'npycam{}.npy'.format(i)), cam)
    
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
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) 
    heatmap = np.float32(heatmap) / (255 + 1e-6)
    cam = heatmap + np.float32(img)
    cam = cam / (np.max(cam) + 1e-6)
    return cam, heatmap, img


def calc_saliency_maps(model, inputs, tabular_data, config, device, c):
    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)

    if config['model']['backbone'] == 'r2plus1_18':
        layer_name = 'model.module.encoder.4.1.relu'
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
        
    elif config['model']['backbone'] == 'swin_s':
        if config['is_already_trained']:
            layer_name = "model.module.encoder.features.6.1.norm1"
        else:
            layer_name = "model.module.encoder.features.6.1.norm1"    
    elif config['model']['backbone'] == 'swin_tiny':
        if config['is_already_trained']:
            layer_name = "model.module.encoder.features.6.1.norm1"
        else:
            layer_name = "model.module.encoder.features.6.1.norm1"

    else:
        raise ValueError('not implemented')
    # if config['model']['backbone'] not in ['r2plus1_18', 'x3d_s', 'debug_3d']:

    #     target_layers = [model.module.encoder.features[-1][-1].norm1]
    #     class ModelWithTabularData(torch.nn.Module):
    #         def __init__(self, model, tabular_data):
    #             super(ModelWithTabularData, self).__init__()
    #             self.model = model
    #             self.tabular_data = tabular_data

    #         def forward(self, x):
    #             return self.model(x, tabular_data=self.tabular_data)
    #     cam_f = GradCAM(
    #         model=model,
    #         target_layers=target_layers,
    #                 reshape_transform=reshape_transform)

    #     # targets = [ClassifierOutputTarget(0)]
        
    #     cam = cam_f(input_tensor=inputs, tabular_data=tabular_data)
    #     cam = resize3dVolume(
    #         cam[0, :, :, :],
    #         (config['loaders']['Crop_height'],
    #             config['loaders']['Crop_width'],
    #             config['loaders']['Crop_depth']))
    #     cam = np.expand_dims(np.expand_dims(cam, 0), 0)

    # else:
    from monai.visualize import GradCAMpp, OcclusionSensitivity
    from miacag.utils.GradCam_j import GradCAMj, ClassifierOutputTarget
    # saliency = SaliencyInferer(
    #     cam_name="GradCAM",
    #     target_layers=layer_name,
    #     class_idx=c)
    model.eval()
    class ModelWithTabularData(torch.nn.Module):
        def __init__(self, model, tabular_data):
            super(ModelWithTabularData, self).__init__()
            self.model = model
            self.tabular_data = tabular_data

        def forward(self, x):
            return self.model(x, tabular_data=self.tabular_data)
    model_with_tabular = ModelWithTabularData(model, tabular_data)
    
    model_with_tabular.eval()
    # gradcam = GradCAMR2D(model_with_tabular, target_layers=layer_name)
    # cam = gradcam(inputs, class_idx=c).cpu() #, retain_graph=True)
    from miacag.model_utils.GradCam_model import GradCAM as GradCAM_old
    camj = GradCAM_old(model=model_with_tabular, target_layers=[model_with_tabular.model.module.encoder[-1]])
    targets = [ClassifierOutputTarget(c)]
    cam =  camj(input_tensor=inputs, targets=targets)
    cam = np.rollaxis(np.rollaxis(np.squeeze(cam), 1,0), 2,1)
    cam = resize3dVolume(np.squeeze(cam), (inputs.shape[2], inputs.shape[3], inputs.shape[4]))
    cam = np.expand_dims(np.expand_dims(cam, 0), 0)
    cam = torch.from_numpy(cam).float()



    if config['model']['dimension'] in ['2D+T', "3D"]:
        return cam[0:1, :, :, :, :]
    elif config['model']['dimension'] in ['2D']:
        return cam[0:1, :, :, :]
    else:
        raise ValueError('not implemented')


# def reshape_transform(tensor, height=14, width=14):ra
#     tensor = tensor[:, 1:, :]
#     tensor = tensor.unsqueeze(dim=0)
#     result = torch.nn.functional.interpolate(
#         tensor,
#         scale_factor=(392/tensor.size(2), 1))
#     result = result.reshape(result.size(0), 8, 7, 7, result.size(-1))
#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.permute(0, 4, 1, 2, 3)
#     return result

def reshape_transform(tensor, height=4, width=4, depth=2):
    result = tensor.reshape(tensor.size(0),
         8, 4, 4, tensor.size(4))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result =result.permute(0, 4, 1, 2, 3)
    return result
def prepare_model_for_sm(model, config, c):
    if config['task_type'] in ['regression', 'classification']:
       # copy_model = copy.deepcopy(model)
       # copy_model.module.module.fc = copy_model.module.module.fcs[c]
        bound_method = model.module.forward_saliency.__get__(
            model, model.module.__class__)
        setattr(model.module, 'forward', bound_method)
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
