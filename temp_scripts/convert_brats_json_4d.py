import os
import glob
import pandas as pd
import SimpleITK as sitk
import numpy as np


def get_files(path, phase_list, out_path_image, file_format):
    flair = []
    t1 = []
    t1ce = []
    seg = []
    t2 = []
    name = []
    mod1id = []
    mod2id = []
    mod3id = []
    mod4id = []
    flairmod = []
    t1mod = []
    t1cemod = []
    t2mod = []
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

        filename = convert2mnc_and_stack(path, flair_path, t1_path, t1ce_path, t2_path, seg_path, out_path_image, file_format)
        new = ['0001', '0002', '0003', '0004']
        old = ['flair', 't1', 't1ce', 't2']
        # flair_path, flair_mod = convert2mnc(path, flair_path, out_path_image, file_format, new[0])
        # t1_path, t1_mod = convert2mnc(path, t1_path, out_path_image, file_format, new[1])
        # t1ce_path, t1ce_mod = convert2mnc(path, t1ce_path, out_path_image, file_format, new[2])
        # t2_path, t2_mod = convert2mnc(path, t2_path, out_path_image, file_format, new[3])
        # seg_path, _ = convert2mnc(path, seg_path, out_path_image, file_format, 'seg')

        mod1id.append(new[0])
        mod2id.append(new[1])
        mod3id.append(new[2])
        mod4id.append(new[3])

        flairmod.append(old[0])
        t1mod.append(old[1])
        t1cemod.append(old[2])
        t2mod.append(old[3])

        flair.append(flair_path)
        t1.append(t1_path)
        t1ce.append(t1ce_path)
        seg.append(seg_path)
        t2.append(t2_path)
        name.append(os.listdir(os.path.join(path, phase_list[i]))[0][0:20])

    df = pd.DataFrame(list(zip(name, flair, flairmod, mod1id,
                               t1, t1mod, mod2id,
                               t1ce, t1cemod, mod3id,
                               t2, t2mod, mod4id, seg)),
                      columns=['name', 'image1', 'mod1', 'mod1id',
                               'image2', 'mod2', 'mod2id',
                               'image3', 'mod3', 'mod3id',
                               'image4', 'mod4', 'mod4id',
                               'seg'])
    return df


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert2mnc(path, mod_path, outfile, file_format, new_mod):
    outfile_leaf = os.path.splitext(os.path.splitext(mod_path)[0])[0]
    if new_mod != 'seg':
        new_path = ''.join(outfile_leaf.rsplit('_', 1)[0]+'_'+new_mod)
        old_mod = outfile_leaf.rsplit('_', 1)[1]
    else:
        new_path = outfile_leaf
        old_mod = None
    mkDir(os.path.join(outfile, os.path.dirname(new_path)))
    image = sitk.ReadImage(os.path.join(path, mod_path))
    sitk.WriteImage(image, os.path.join(outfile, new_path+file_format))
    minc_filename = os.path.join(os.path.basename(outfile),
                                 new_path+file_format)
    return minc_filename, old_mod


def convert2mnc_and_stack(path, mod1_path, mod2_path, mod3_path, mod4_path, seg_path, outfile, file_format):
    outfile_leaf = os.path.dirname(mod1_path)
    image1 = sitk.ReadImage(os.path.join(path, mod1_path))
    image2 = sitk.ReadImage(os.path.join(path, mod2_path))
    image3 = sitk.ReadImage(os.path.join(path, mod3_path))
    image4 = sitk.ReadImage(os.path.join(path, mod4_path))
    seg = sitk.ReadImage(os.path.join(path, seg_path))
    numpyOrigin = (0.0,) + image1.GetOrigin()
    numpySpacing = (1.0,) + image1.GetSpacing()
    array1 = np.expand_dims(sitk.GetArrayFromImage(image1), 0)
    array2 = np.expand_dims(sitk.GetArrayFromImage(image2), 0)
    array3 = np.expand_dims(sitk.GetArrayFromImage(image3), 0)
    array4 = np.expand_dims(sitk.GetArrayFromImage(image4), 0)
    array = np.concatenate((array1, array2, array3, array4), axis=0)
    image = sitk.GetImageFromArray(array, isVector=False)
    image.SetSpacing(numpySpacing)
    image.SetOrigin(numpySpacing)
    mkDir(os.path.join(outfile, outfile_leaf))
   # image = sitk.ReadImage(os.path.join(path, mod_path))
    minc_filename = os.path.join(outfile, outfile_leaf, outfile_leaf +file_format)
    sitk.WriteImage(image, minc_filename)
    sitk.WriteImage(seg, os.path.join(outfile, outfile_leaf, outfile_leaf +'_seg_' + file_format))
    out_minc = os.path.join(outfile_leaf, outfile_leaf +file_format)
    out_seg = os.path.join(outfile_leaf, outfile_leaf +'_seg_' + file_format)
    return out_minc, out_seg

path = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/image_data_raw'
meta_path = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/survival_info.csv'
out_path_image = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/image_data_processed'

out_train_csv = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/train.csv'
out_val_csv = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/val.csv'
out_test_csv = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/test.csv'
file_format = '.nii.gz'

mkDir(out_path_image)
folders = os.listdir(path)
nr_sujbects = len(folders)
train = folders[:int(nr_sujbects*0.7)]
val = folders[int(nr_sujbects*0.7):int(nr_sujbects*(0.7+0.15))]
test = folders[int(nr_sujbects*(0.7+0.15)):]

train_df = get_files(path, train, out_path_image, file_format)
val_df = get_files(path, val, out_path_image, file_format)
test_df = get_files(path, test, out_path_image, file_format)

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
train_df.to_json(out_train_csv[:-4]+'.json')

val_df.to_csv(out_val_csv, index=False)
val_df.to_json(out_val_csv[:-4]+'.json')

test_df.to_csv(out_test_csv, index=False)
test_df.to_json(out_test_csv[:-4]+'.json')

#  Graveyard  ###
# def get_files(path, phase_list, out_path_image, file_format):
#     flair = []
#     t1 = []
#     t1ce = []
#     seg = []
#     t2 = []
#     name = []
#     mod1id = []
#     mod2id = []
#     mod3id = []
#     mod4id = []
#     flairmod = []
#     t1mod = []
#     t1cemod = []
#     t2mod = []
#     for i in range(0, len(phase_list)):
#         dir_path = os.path.join(path, phase_list[i])

#         flair_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path)
#                       if x.endswith('flair.nii.gz')][0]
#         t1_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path) 
#                       if x.endswith('t1.nii.gz')][0]
#         t1ce_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path) 
#                       if x.endswith('t1ce.nii.gz')][0]
#         t2_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path) 
#                       if x.endswith('t2.nii.gz')][0]
#         seg_path = [os.path.join(phase_list[i], x) for x in os.listdir(dir_path) 
#                       if x.endswith('seg.nii.gz')][0]

#         new = ['0001', '0002', '0003', '0004']
#         old = ['flair', 't1', 't1ce', 't2']
#         flair_path, flair_mod = convert2mnc(path, flair_path, out_path_image, file_format, new[0])
#         t1_path, t1_mod = convert2mnc(path, t1_path, out_path_image, file_format, new[1])
#         t1ce_path, t1ce_mod = convert2mnc(path, t1ce_path, out_path_image, file_format, new[2])
#         t2_path, t2_mod = convert2mnc(path, t2_path, out_path_image, file_format, new[3])
#         seg_path, _ = convert2mnc(path, seg_path, out_path_image, file_format, 'seg')

#         mod1id.append(new[0])
#         mod2id.append(new[1])
#         mod3id.append(new[2])
#         mod4id.append(new[3])

#         flairmod.append(flair_mod)
#         t1mod.append(t1_mod)
#         t1cemod.append(t1ce_mod)
#         t2mod.append(t2_mod)

#         flair.append(flair_path)
#         t1.append(t1_path)
#         t1ce.append(t1ce_path)
#         seg.append(seg_path)
#         t2.append(t2_path)
#         name.append(os.listdir(os.path.join(path, phase_list[i]))[0][0:20])

#     df = pd.DataFrame(list(zip(name, flair, flairmod, mod1id,
#                                t1, t1mod, mod2id,
#                                t1ce, t1cemod, mod3id,
#                                t2, t2mod, mod4id, seg)),
#                       columns=['name', 'image1', 'mod1', 'mod1id',
#                                'image2', 'mod2', 'mod2id',
#                                'image3', 'mod3', 'mod3id',
#                                'image4', 'mod4', 'mod4id',
#                                'seg'])
#     return df