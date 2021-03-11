import av
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
import nibabel as nib


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

path = '/home/gandalf/mmaction2/kinetics400_tiny/train'
outpath = '/home/gandalf/segmento/data/kinetics400_tiny_train_nifty'
train_list = '/home/gandalf/mmaction2/kinetics400_tiny/kinetics_tiny_train_video.txt'
text_file = open(train_list, "r")
lines = text_file.readlines()
data = pd.read_csv(train_list, sep=" ", header=None)

vid_paths = []
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

    for channel in range(0, array_3d.shape[2]):
        arr = np.expand_dims(array_3d[:, :, channel, :], -1)
        arr = np.transpose(arr, (0, 1, 3, 2))
        img = nib.Nifti1Image(arr, np.eye(4))
        channel_path = vid_path[:-4] + '_000' + str(channel+1) + '.nii.gz'
        vid_paths.append(channel_path)
        vid_labels.append(vid_label)
        nib.save(img, os.path.join(outpath, channel_path))

df = pd.DataFrame(vid_labels, vid_paths)
df.to_csv('data/kinetics_tiny_minc.csv')
print('done')