import av
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
import nibabel as nib
import json


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

path = '/home/gandalf/mmaction2/kinetics400_tiny/train'
outpath = '/home/gandalf/segmento/data/kinetics400_tiny_train_nifty/data/'
train_list = '/home/gandalf/mmaction2/kinetics400_tiny/kinetics_tiny_train_video.txt'
json_out_path = '/home/gandalf/segmento/data/kinetics400_tiny_train_nifty/data.json'
text_file = open(train_list, "r")
lines = text_file.readlines()
data = pd.read_csv(train_list, sep=" ", header=None)

vid_paths_r = []
vid_paths_g = []
vid_paths_b = []
vid_labels = []
for f in range(0, len(data)):
    vid_path = data.iloc[f, 0]
    vid_label = data.iloc[f, 1]
    mkDir(os.path.dirname(os.path.join(outpath, vid_path[:-4])))

    container = av.open(os.path.join(path, vid_path))
    array_3d = np.zeros((
                        container.streams.video[0].height,
                        container.streams.video[0].width,
                        container.streams.video[0].frames, 3), dtype=np.uint8)
    i = 0
    for frame in container.decode(video=0):
        array_2d = np.array(frame.to_image())
        array_3d[:, :, i, :] = array_2d  # np.rollaxis(array_2d,2)
        i += 1
    array_3d = np.transpose(array_3d, (0, 1, 3, 2))

#     #
#     img = nib.Nifti1Image(array_3d, np.eye(4).astype('uint8'))
#   #  img.get_data_dtype() == np.dtype(np.uint8)
#     img.header.set_data_dtype(np.uint8)
#    # img.set_data_dtype(np.uint8)
#     vid_labels.append(vid_label)
#     channel_path = vid_path[:-4] + '.nii.gz'
#     nib.save(img, os.path.join(outpath, channel_path))
#     sitk_a = np.expand_dims(array_3d, -1)
#     sitk_a = np.transpose(sitk_a, (3, 4, 1, 0, 2))
#     img_istk = sitk.GetImageFromArray(sitk_a, isVector=True)
#     sitk.WriteImage(img_istk,
#                     os.path.join(outpath, vid_path[:-4] + 'sitk.nii.gz'))

#     img_istk_0 = sitk.GetImageFromArray(sitk_a[:,:,:,: 0], isVector=True)
#     sitk.WriteImage(img_istk,
#                     os.path.join(outpath, vid_path[:-4] + 'sitk.nii.gz'))
#     vid_paths_r.append(channel_path)
    #
    for channel in range(0, array_3d.shape[2]):
        arr = np.expand_dims(array_3d[:, :, channel, :], -1)
        arr = np.transpose(arr, (0, 1, 3, 2))
        img = nib.Nifti1Image(arr, np.eye(4))
        channel_path = vid_path[:-4] + '_000' + str(channel+1) + '.nii.gz'
        if channel == 0:
            vid_paths_r.append(channel_path)
        elif channel == 1:
            vid_paths_g.append(channel_path)
        if channel == 2:
            vid_paths_b.append(channel_path)
        vid_labels.append(vid_label)
        nib.save(img, os.path.join(outpath, channel_path))
vid_labels = [int(lab) for lab in vid_labels]
vid_labels_dict = {'labels': vid_labels}
modalities = {'r': '0001', 'g': '0002', 'b': '0003'}
vid_paths_r_dict = {'r_paths': vid_paths_r}
vid_paths_g_dict = {'g_paths': vid_paths_g}
vid_paths_b_dict = {'b_paths': vid_paths_b}

dict_json = {**modalities, **vid_labels_dict,
             **vid_paths_r_dict, **vid_paths_g_dict, **vid_paths_b_dict}
with open(json_out_path, 'w') as file:
    file.write(json.dumps(dict_json,
                          sort_keys=True, indent=4,
                          separators=(',', ': ')))
df = pd.DataFrame(list(zip(vid_paths_r, vid_paths_g,
                  vid_paths_b, vid_labels)),
                  columns=['image1', 'image2', 'image3', 'labels'])
df.to_csv(json_out_path[0:-5]+'.csv')
print('done')