import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from metrics.metrics_utils import mkDir


def normalize(img):
    img = (img - np.min(img)) / \
        (np.amax(img)-np.amin(img)) # + 1e-8
    return img

def prepare_cv2_img(img, mask, patientID,
                    studyInstanceUID,
                    seriesInstanceUID,
                    SOPInstanceUID,
                    config):
    path = os.path.join(
        config['output_directory'],
        'saliency',
        'mispredictions'
        if config['loaders']['val_method']['misprediction'] == 'True'
        else 'corrects',
        patientID,
        studyInstanceUID,
        seriesInstanceUID,
        SOPInstanceUID)
    mkDir(path)
    img = img[0, 0, :, :, :]
    img = np.expand_dims(img, 2)
    mask = mask[0, 0, :, :, :]
    mask = np.expand_dims(mask, 2)
    for i in range(0, img.shape[3]):
        input2d = img[:, :, :, i]
        cam, heatmap, input2d = show_cam_on_image(
            input2d,
            mask[:, :, :, i])

        input2d = np.flip(np.rot90(np.rot90(np.rot90(input2d))), 1)
        plt.imshow(input2d, cmap="gray", interpolation="None")
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'input{}.png'.format(i)))
        plt.clf()

        cam = np.flip(np.rot90(np.rot90(np.rot90(cam))), 1)
        plt.imshow(cam, cmap="jet", interpolation="None")
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'cam{}.png'.format(i)))
        plt.clf()

        heatmap = np.flip(np.rot90(np.rot90(np.rot90(heatmap))), 1)
        plt.imshow(heatmap, cmap="jet", interpolation="None")
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'heatmap{}.png'.format(i)))
        plt.clf()

def show_cam_on_image(img, mask):
    img = normalize(img)
    mask = normalize(mask)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam, heatmap, img
