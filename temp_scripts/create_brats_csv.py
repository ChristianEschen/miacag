import os
import glob
import pandas as pd
import SimpleITK as sitk

path = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/image_data_raw'
meta_path = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/survival_info.csv'
out_path_image = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/image_data_mnc'

out_train_csv = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/train.csv'
out_val_csv = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/val.csv'
out_test_csv = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/test.csv'

folders = os.listdir(path)
nr_sujbects = len(folders)
train = folders[:int(nr_sujbects*0.7)]
val = folders[int(nr_sujbects*0.7):int(nr_sujbects*(0.7+0.15))]
test = folders[int(nr_sujbects*(0.7+0.15)):]


def get_files(path, phase_list, out_path_image):
    flair = []
    t1 = []
    t1ce = []
    seg = []
    t2 = []
    name = []
    for i in range(0, len(phase_list)):
        dir_path = os.path.join(path, phase_list[i])

        flair_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path)
                      if x.endswith('flair.nii.gz')][0]
        t1_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path) 
                      if x.endswith('t1.nii.gz')][0]
        t1ce_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path) 
                      if x.endswith('t1ce.nii.gz')][0]
        t2_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path) 
                      if x.endswith('t2.nii.gz')][0]
        seg_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path) 
                      if x.endswith('seg.nii.gz')][0]

        flair_path = convert2mnc(path, flair_path, out_path_image)
        t1_path = convert2mnc(path, t1_path, out_path_image)
        t1ce_path = convert2mnc(path, t1ce_path, out_path_image)
        t2_path = convert2mnc(path, t2_path, out_path_image)
        seg_path = convert2mnc(path, seg_path, out_path_image)

        flair.append(flair_path)
        t1.append(t1_path)
        t1ce.append(t1ce_path)
        seg.append(seg_path)
        t2.append(t2_path)
        name.append(os.listdir(os.path.join(path, phase_list[i]))[0][0:20])

    df = pd.DataFrame(list(zip(name, flair, t1, t1ce, t2, seg)),
                      columns=['name', 'flair', 't1', 't1ce', 't2', 'seg'])
    return df


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert2mnc(path, mod_path, outfile):
    outfile_leaf = os.path.splitext(os.path.splitext(mod_path)[0])[0]
    mkDir(os.path.join(outfile, os.path.dirname(outfile_leaf)))
    image = sitk.ReadImage(os.path.join(path, mod_path))
    sitk.WriteImage(image, os.path.join(outfile, outfile_leaf+'.mnc'))
    minc_filename = os.path.join(os.path.basename(outfile),
                                 outfile_leaf+'.mnc')
    return minc_filename

train_df = get_files(path, train, out_path_image)
val_df = get_files(path, val, out_path_image)
test_df = get_files(path, test, out_path_image)

meta_df = pd.read_csv(meta_path)

train_df = train_df.merge(meta_df,
                          left_on='name',
                          right_on='Brats20ID')

val_df = val_df.merge(meta_df,
                      left_on='name',
                      right_on='Brats20ID')

test_df = test_df.merge(meta_df,
                        left_on='name',
                        right_on='Brats20ID')

train_df.to_csv(out_train_csv, index=False)

val_df.to_csv(out_val_csv, index=False)

test_df.to_csv(out_test_csv, index=False)