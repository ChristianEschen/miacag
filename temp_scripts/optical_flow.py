import cv2
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import imageio
import os
from PIL import Image
import pandas as pd


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_video(path, path1, path2):
    img1 = nib.load(path)
    array1 = img1.get_fdata()
    img2 = nib.load(path1)
    array2 = img2.get_fdata()
    img3 = nib.load(path2)
    array3 = img3.get_fdata()
    array = np.concatenate((array1, array2, array3), axis=2)
    return array, img1

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def computeFarneback(array_i, array_i_plus, hsv):
    flow_i = cv2.calcOpticalFlowFarneback(array_i, array_i_plus,
                                          None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow_i[:, :, 0], flow_i[:, :, 1])

    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = np.uint8(hsv)
    flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow


def computedualTVL1_Flow(array_i, array_i_plus):
    dualTV = cv2.optflow.DualTVL1OpticalFlow_create()
    flow_i = dualTV.calc(np.uint8(array_i), np.uint8(array_i_plus), None)
    return flow_i

def normalize(img2d):
    img2d = (img2d - np.amin(img2d)) / (
            np.amax(img2d)-np.amin(img2d)) * 255
    return img2d
def save_gif(img3d):
    gif= []
    for i in range(0, img3d.shape[0]):
        img2d = np.squeeze(img3d[i, :, :, :]).astype('float32')
        gif.append(Image.fromarray(np.uint8(img2d)))
    gif[0].save('gif.gif', save_all=True,
                optimize=False, append_images=gif[1:])

def reflectpad(flow, video):
    flow_pad = np.zeros_like(video)
    diff = video.shape[0] - flow.shape[0]
    flow_last = np.repeat(flow[-1], diff, axis=0)
    if flow.shape[0] < video.shape[0]:
        flow_pad[0:video.shape[0]-diff, :, :, :] = flow
        flow_pad[video.shape[0]-diff:] = flow_last
    return flow_pad

def computeOpticalFlow(array, f_type='Farne'): 
    flow_gray = np.zeros((array.shape[0]-1, array.shape[1], array.shape[2], 3))
    flow_gray = flow_gray.astype('uint8')
    hsv = np.zeros((array.shape[1], array.shape[2], 3))
    hsv[..., 2] = 0


    for i in range(0, array.shape[0]-1):
        #####
        grayscale = cv2.cvtColor(np.uint8(array[i, :, :, :]), cv2.COLOR_RGB2GRAY)
        grayscale_next = cv2.cvtColor(np.uint8(array[i+1, :, :, :]), cv2.COLOR_RGB2GRAY)
        if f_type == 'Farne':
            flow_i = computeFarneback(grayscale, grayscale_next, hsv)
        elif f_type == 'dualTV':
            flow_i = computedualTVL1_Flow(grayscale, grayscale_next)
            hsv[..., 0] = normalize(flow_i[:, :, 0])
            hsv[..., 1] = normalize(flow_i[:, :, 1])
            flow_i = hsv
        else:
            print('WRONG flow type')
        data = np.uint8(flow_i)
        flow_gray[i, :, :, :] = flow_i
    return flow_gray

def main():
    csv_path = '/home/gandalf/MIA/data/kinetics400_tiny_train_nifty/data.csv'
    out_csv_path = '/home/gandalf/MIA/data/kinetics400_tiny_train_nifty/data_flow.csv'
    data_path = '/home/gandalf/MIA/data/kinetics400_tiny_train_nifty/data'
    flow_path = '/home/gandalf/MIA/data/kinetics400_tiny_train_nifty/flow'
    mkDir(flow_path)
    df = pd.read_csv(csv_path)
    outnames0 = []
    outnames1 = []
    for index, row in df.iterrows():
        print('index', index)
        array, img = load_video(os.path.join(data_path, row['image1']),
                                os.path.join(data_path, row['image2']),
                                os.path.join(data_path, row['image3']))
        array_reshaped = np.transpose(np.squeeze(array), (3, 0, 1, 2))
        optic_array = computeOpticalFlow(array_reshaped, 'dualTV')
        optic_array = reflectpad(optic_array, array_reshaped)
        # imageio.mimwrite('output_filename_' + f_type + '.mp4',
        #                 optic_array, fps = 29.970030)

        optic_array = np.uint8(np.transpose(optic_array, (1, 2, 3, 0)))

        optic_image0 = nib.Nifti1Image(
            optic_array[:, :, 0, :], img.affine, header=img.header)
        optic_image1 = nib.Nifti1Image(
            optic_array[:, :, 1, :], img.affine, header=img.header)
        outname0 = os.path.join(data_path, row['image1'])[:-11]+'flow0.nii.gz'
        outname1 = os.path.join(data_path, row['image1'])[:-11]+'flow1.nii.gz'
        nib.save(
            optic_image0,
            outname0)
        nib.save(
            optic_image1,
            outname1)

        outnames0.append(os.path.basename(outname0))
        outnames1.append(os.path.basename(outname1))

    df_flow = pd.DataFrame(
        list(zip(outnames0, outnames1)),
        columns =['flow0', 'flow1'])
    df = pd.concat([df, df_flow], axis=1)
    df.to_csv('/home/gandalf/MIA/data/kinetics400_tiny_train_nifty/data_flow.csv', index=False)
    print('done')

    
if __name__ == '__main__':
    main()
