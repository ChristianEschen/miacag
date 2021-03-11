#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from PIL import Image
import shutil
from collections import OrderedDict
import pandas as pd
import dicom2nifti
import nibabel as nib
import numpy as np
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
from PIL import Image
import SimpleITK as sitk
from nnunet.paths import preprocessing_output_dir, nnUNet_raw_data
from nnunet.utilities.sitk_stuff import copy_geometry
from nnunet.inference.ensemble_predictions import merge


def getKeysFromDict(dictionary, search_cat):
    for cat, mod in dictionary.items():
        if cat == search_cat:
            return mod

def load_png_stack(folder):
    pngs = subfiles(folder, suffix="png")
    pngs.sort()
    loaded = []
    for p in pngs:
        loaded.append(np.array(Image.open(p)))
    loaded = np.stack(loaded, 0)[::-1]
    return loaded


def convert_CT_seg(loaded_png):
    return loaded_png.astype(np.uint16)


def convert_MR_seg(loaded_png):
    result = np.zeros(loaded_png.shape)
    result[(loaded_png > 55) & (loaded_png <= 70)] = 1 # liver
    result[(loaded_png > 110) & (loaded_png <= 135)] = 2 # right kidney
    result[(loaded_png > 175) & (loaded_png <= 200)] = 3 # left kidney
    result[(loaded_png > 240) & (loaded_png <= 255)] = 4 # spleen
    return result


def convert_seg_to_intensity_task5(seg):
    seg_new = np.zeros(seg.shape, dtype=np.uint8)
    seg_new[seg == 1] = 63
    seg_new[seg == 2] = 126
    seg_new[seg == 3] = 189
    seg_new[seg == 4] = 252
    return seg_new


def convert_seg_to_intensity_task3(seg):
    seg_new = np.zeros(seg.shape, dtype=np.uint8)
    seg_new[seg == 1] = 63
    return seg_new


def write_pngs_from_nifti(nifti, output_folder, converter=convert_seg_to_intensity_task3):
    npy = sitk.GetArrayFromImage(sitk.ReadImage(nifti))
    seg_new = converter(npy)
    for z in range(len(npy)):
        Image.fromarray(seg_new[z]).save(join(output_folder, "img%03.0d.png" % z))


def convert_variant2_predicted_test_to_submission_format(folder_with_predictions,
                                                         output_folder="/home/fabian/drives/datasets/results/nnUNet/test_sets/Task038_CHAOS_Task_3_5_Variant2/ready_to_submit",
                                                         postprocessing_file="/home/fabian/drives/datasets/results/nnUNet/ensembles/Task038_CHAOS_Task_3_5_Variant2/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json"):
    """
    output_folder is where the extracted template is
    :param folder_with_predictions:
    :param output_folder:
    :return:
    """
    postprocessing_file = "/media/fabian/Results/nnUNet/3d_fullres/Task039_CHAOS_Task_3_5_Variant2_highres/" \
                          "nnUNetTrainerV2__nnUNetPlansfixed/postprocessing.json"

    # variant 2 treats in and out phase as two training examples, so we need to ensemble these two again
    final_predictions_folder = join(output_folder, "final")
    maybe_mkdir_p(final_predictions_folder)
    t1_patient_names = [i.split("_")[-1][:-7] for i in subfiles(folder_with_predictions, prefix="T1", suffix=".nii.gz", join=False)]
    folder_for_ensembing0 = join(output_folder, "ens0")
    folder_for_ensembing1 = join(output_folder, "ens1")
    maybe_mkdir_p(folder_for_ensembing0)
    maybe_mkdir_p(folder_for_ensembing1)
    # now copy all t1 out phases in ens0 and all in phases in ens1. Name them the same.
    for t1 in t1_patient_names:
        shutil.copy(join(folder_with_predictions, "T1_in_%s.npz" % t1), join(folder_for_ensembing1, "T1_%s.npz" % t1))
        shutil.copy(join(folder_with_predictions, "T1_in_%s.pkl" % t1), join(folder_for_ensembing1, "T1_%s.pkl" % t1))
        shutil.copy(join(folder_with_predictions, "T1_out_%s.npz" % t1), join(folder_for_ensembing0, "T1_%s.npz" % t1))
        shutil.copy(join(folder_with_predictions, "T1_out_%s.pkl" % t1), join(folder_for_ensembing0, "T1_%s.pkl" % t1))
    shutil.copy(join(folder_with_predictions, "plans.pkl"), join(folder_for_ensembing0, "plans.pkl"))
    shutil.copy(join(folder_with_predictions, "plans.pkl"), join(folder_for_ensembing1, "plans.pkl"))

    # there is a problem with T1_35 that I need to correct manually (different crop size, will not negatively impact results)
    #ens0_softmax = np.load(join(folder_for_ensembing0, "T1_35.npz"))['softmax']
    ens1_softmax = np.load(join(folder_for_ensembing1, "T1_35.npz"))['softmax']
    #ens0_props = load_pickle(join(folder_for_ensembing0, "T1_35.pkl"))
    #ens1_props = load_pickle(join(folder_for_ensembing1, "T1_35.pkl"))
    ens1_softmax = ens1_softmax[:, :, :-1, :]
    np.savez_compressed(join(folder_for_ensembing1, "T1_35.npz"), softmax=ens1_softmax)
    shutil.copy(join(folder_for_ensembing0, "T1_35.pkl"), join(folder_for_ensembing1, "T1_35.pkl"))

    # now call my ensemble function
    merge((folder_for_ensembing0, folder_for_ensembing1), final_predictions_folder, 8, True,
          postprocessing_file=postprocessing_file)
    # copy t2 files to final_predictions_folder as well
    t2_files = subfiles(folder_with_predictions, prefix="T2", suffix=".nii.gz", join=False)
    for t2 in t2_files:
        shutil.copy(join(folder_with_predictions, t2), join(final_predictions_folder, t2))

    # apply postprocessing
    from nnunet.postprocessing.connected_components import apply_postprocessing_to_folder, load_postprocessing
    postprocessed_folder = join(output_folder, "final_postprocessed")
    for_which_classes, min_valid_obj_size = load_postprocessing(postprocessing_file)
    apply_postprocessing_to_folder(final_predictions_folder, postprocessed_folder,
                                   for_which_classes, min_valid_obj_size, 8)

    # now export the niftis in the weird png format
    # task 3
    output_dir = join(output_folder, "CHAOS_submission_template_new", "Task3", "MR")
    for t1 in t1_patient_names:
        output_folder_here = join(output_dir, t1, "T1DUAL", "Results")
        nifti_file = join(postprocessed_folder, "T1_%s.nii.gz" % t1)
        write_pngs_from_nifti(nifti_file, output_folder_here, converter=convert_seg_to_intensity_task3)
    for t2 in t2_files:
        patname = t2.split("_")[-1][:-7]
        output_folder_here = join(output_dir, patname, "T2SPIR", "Results")
        nifti_file = join(postprocessed_folder, "T2_%s.nii.gz" % patname)
        write_pngs_from_nifti(nifti_file, output_folder_here, converter=convert_seg_to_intensity_task3)

    # task 5
    output_dir = join(output_folder, "CHAOS_submission_template_new", "Task5", "MR")
    for t1 in t1_patient_names:
        output_folder_here = join(output_dir, t1, "T1DUAL", "Results")
        nifti_file = join(postprocessed_folder, "T1_%s.nii.gz" % t1)
        write_pngs_from_nifti(nifti_file, output_folder_here, converter=convert_seg_to_intensity_task5)
    for t2 in t2_files:
        patname = t2.split("_")[-1][:-7]
        output_folder_here = join(output_dir, patname, "T2SPIR", "Results")
        nifti_file = join(postprocessed_folder, "T2_%s.nii.gz" % patname)
        write_pngs_from_nifti(nifti_file, output_folder_here, converter=convert_seg_to_intensity_task5)



if __name__ == "__main__":
    """
    This script only prepares data to participate in Task 5 and Task 5. I don't like the CT task because 
    1) there are 
    no abdominal organs in the ground truth. In the case of CT we are supposed to train only liver while on MRI we are 
    supposed to train all organs. This would require manual modification of nnU-net to deal with this dataset. This is 
    not what nnU-net is about.
    2) CT Liver or multiorgan segmentation is too easy to get external data for. Therefore the challenges comes down 
    to who gets the b est external data, not who has the best algorithm. Not super interesting.
    
    Task 3 is a subtask of Task 5 so we need to prepare the data only once.
    Difficulty: We need to process both T1 and T2, but T1 has 2 'modalities' (phases). nnU-Net cannot handly varying 
    number of input channels. We need to be creative.
    We deal with this by preparing 2 Variants:
    1) pretend we have 2 modalities for T2 as well by simply stacking a copy of the data
    2) treat all MRI sequences independently, so we now have 3*20 training data instead of 2*20. In inference we then 
    ensemble the results for the two t1 modalities. 
    
    Careful: We need to split manually here to ensure we stratify by patient
    """

    root = "/media/gandalf/AE3416073415D2E7/CHAOS_pancreas_MRI/Train_Sets"
    root_test = "/media/gandalf/AE3416073415D2E7/CHAOS_pancreas_MRI/Test_Sets"
    out_base = "/media/gandalf/AE3416073415D2E7/CHAOS_pancreas_MRI_nifty"
    # CT
    # we ignore CT because


    ##############################################################
    # Variant 2
    ##############################################################

    patient_ids = []
    patient_ids_val = []
    patient_ids_test = []

    task_name = "Task038_CHAOS_Task_3_5_Variant2"
    output_folder = join(out_base, task_name)
    output_images = join(output_folder, "imagesTr")
    output_labels = join(output_folder, "labelsTr")
    output_images_val = join(output_folder, "imagesVal")
    output_labels_val = join(output_folder, "labelsVal")
    output_imagesTs = join(output_folder, "imagesTs")
    maybe_mkdir_p(output_images)
    maybe_mkdir_p(output_labels)
    maybe_mkdir_p(output_images_val)
    maybe_mkdir_p(output_labels_val)
    maybe_mkdir_p(output_imagesTs)

    # Process T1 train
    d = join(root, "MR")
    patients = subdirs(d, join=False)
    patients = patients[0:int(0.7*len(patients))]
    for p in patients:
        patient_name_in = "T1_in_" + p
        patient_name_out = "T1_out_" + p
        gt_dir = join(d, p, "T1DUAL", "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "InPhase")
        img_outfile = join(output_images, patient_name_in + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "OutPhase")
        img_outfile = join(output_images, patient_name_out + "_0001.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)


        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        sitk.WriteImage(seg_itk, join(output_labels, patient_name_in + ".nii.gz"))
        sitk.WriteImage(seg_itk, join(output_labels, patient_name_out + ".nii.gz"))
        patient_ids.append(patient_name_out)
        patient_ids.append(patient_name_in)

    # Process T1 val
    d = join(root, "MR")
    patients = subdirs(d, join=False)
    patients = patients[int(0.7*len(patients)):]
    for p in patients:
        patient_name_in = "T1_in_" + p
        patient_name_out = "T1_out_" + p
        gt_dir = join(d, p, "T1DUAL", "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "InPhase")
        img_outfile = join(output_images_val, patient_name_in + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "OutPhase")
        img_outfile = join(output_images_val, patient_name_out + "_0001.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        sitk.WriteImage(seg_itk, join(output_labels_val, patient_name_in + ".nii.gz"))
        sitk.WriteImage(seg_itk, join(output_labels_val, patient_name_out + ".nii.gz"))
        patient_ids_val.append(patient_name_out)
        patient_ids_val.append(patient_name_in)

    # Process T1 test
    d = join(root_test, "MR")
    patients = subdirs(d, join=False)
    for p in patients:
        patient_name_in = "T1_in_" + p
        patient_name_out = "T1_out_" + p
        gt_dir = join(d, p, "T1DUAL", "Ground")

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "InPhase")
        img_outfile = join(output_imagesTs, patient_name_in + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "OutPhase")
        img_outfile = join(output_imagesTs, patient_name_out + "_0001.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        patient_ids_test.append(patient_name_out)
        patient_ids_test.append(patient_name_in)

    # Process T2 train
    d = join(root, "MR")
    patients = subdirs(d, join=False)
    patients = patients[:int(0.7*len(patients))]
    for p in patients:
        patient_name = "T2_" + p

        gt_dir = join(d, p, "T2SPIR", "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])

        img_dir = join(d, p, "T2SPIR", "DICOM_anon")
        img_outfile = join(output_images, patient_name + "_0002.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        sitk.WriteImage(seg_itk, join(output_labels, patient_name + ".nii.gz"))
        patient_ids.append(patient_name)

    # Process T2 val
    d = join(root, "MR")
    patients = subdirs(d, join=False)
    patients = patients[int(0.7*len(patients)):]
    for p in patients:
        patient_name = "T2_" + p

        gt_dir = join(d, p, "T2SPIR", "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])

        img_dir = join(d, p, "T2SPIR", "DICOM_anon")
        img_outfile = join(output_images_val, patient_name + "_0002.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        sitk.WriteImage(seg_itk, join(output_labels_val, patient_name + ".nii.gz"))
        patient_ids_val.append(patient_name)

    # Process T2 test
    d = join(root_test, "MR")
    patients = subdirs(d, join=False)
    for p in patients:
        patient_name = "T2_" + p

        gt_dir = join(d, p, "T2SPIR", "Ground")

        img_dir = join(d, p, "T2SPIR", "DICOM_anon")
        img_outfile = join(output_imagesTs, patient_name + "_0002.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        patient_ids_test.append(patient_name)

    json_dict = OrderedDict()
    json_dict['name'] = "Chaos Challenge Task3/5 Variant 2"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "https://chaos.grand-challenge.org/Data/"
    json_dict['licence'] = "see https://chaos.grand-challenge.org/Data/"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1_in",
        "1": "T1_out",
        "2": "T2",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "liver",
        "2": "right kidney",
        "3": "left kidney",
        "4": "spleen",
    }
    json_dict['numTraining'] = len(patient_ids)
    json_dict['numValidation'] = len(patient_ids_val)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_ids]
    json_dict['validation'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_ids_val]
    json_dict['test'] = []

    save_json(json_dict, join(output_folder, "dataset.json"))

    d = join(root, "MR")
    patients = subdirs(d, join=False)
    patients_train = patients[:int(0.7*len(patients))]
    patients_val = patients[int(0.7*len(patients)):]
    image0 = []
    for i in range(0, len(patients_train)):
        image0.append("imagesTr/" + json_dict['modality']['0'] +
                      "_" + patients_train[i] + "_000" + '0.nii.gz')
    image1 = []
    for i in range(0, len(patients_train)):
        image1.append("imagesTr/" + json_dict['modality']['1'] +
                      "_" + patients_train[i] + "_000" + '1.nii.gz')
    image2 = []
    for i in range(0, len(patients_train)):
        image2.append("imagesTr/" + json_dict['modality']['2'] +
                      "_" + patients_train[i] + "_000" + '2.nii.gz')

    label0_t1 = []
    for i in range(0, len(patients_train)):
        label0_t1.append("labelsTr/" + json_dict['modality']['0'] +
                      "_" + patients_train[i] + '.nii.gz')

    label2_t2 = []
    for i in range(0, len(patients_train)):
        label2_t2.append("labelsTr/" + json_dict['modality']['2'] +
                      "_" + patients_train[i] + '.nii.gz')

    images_train = image0+image1+image2
    labels_train = label0_t1 + label0_t1 + label2_t2
    image0_val = []
    for i in range(0, len(patients_val)):
        image0_val.append("imagesVal/" + json_dict['modality']['0'] +
                      "_" + patients_val[i] + "_000" + '0.nii.gz')
    image1_val = []
    for i in range(0, len(patients_val)):
        image1_val.append("imagesVal/" + json_dict['modality']['1'] +
                      "_" + patients_val[i] + "_000" + '1.nii.gz')
    image2_val = []
    for i in range(0, len(patients_val)):
        image2_val.append("imagesVal/" + json_dict['modality']['2'] +
                      "_" + patients_val[i] + "_000" + '2.nii.gz')

    label0_t1_val = []
    for i in range(0, len(patients_val)):
        label0_t1_val.append("labelsVal/" + json_dict['modality']['0'] +
                      "_" + patients_val[i] + '.nii.gz')

    label2_t2_val = []
    for i in range(0, len(patients_val)):
        label2_t2_val.append("labelsVal/" + json_dict['modality']['2'] +
                      "_" + patients_val[i] + '.nii.gz')

    images_val = image0_val + image1_val + image2_val
    labels_val = label0_t1_val + label0_t1_val + label2_t2_val

    spacings_train = []
    sizes_train = []
    for img in images_train:
        path = os.path.join(out_base, task_name, img)
        nib_img = nib.load(path)
        spacing = nib_img.header.get_zooms()
        spacings_train.append(spacing)
        sizes_train.append(nib_img.shape)

    spacings_val = []
    sizes_val = []
    for img in images_val:
        path = os.path.join(out_base, task_name, img)
        nib_img = nib.load(path)
        spacing = nib_img.header.get_zooms()
        spacings_val.append(spacing)
        sizes_val.append(nib_img.shape)

    mod0 = ['T1_in_T1_out_T2']*len(images_train)
    #mod1 = ['T1_out']*len(patients_train)
    df_train = pd.DataFrame(list(zip(images_train, mod0,
                                 labels_train, sizes_train,
                                 spacings_train)),
                            columns=['image1', 'mod1', 'seg',
                                     'sizes',
                                     'spacings'])
    mod0 = ['T1_in_T1_out_T2']*len(images_val)
   # mod0 = ['T1_in']*len(patients_val)
    #mod1 = ['T1_out']*len(patients_val)
    df_val = pd.DataFrame(list(zip(images_val, mod0,
                                   labels_val,
                                   sizes_val, spacings_val)),
                          columns=['image1', 'mod1',
                                   'seg',
                                   'sizes', 'spacings'])

    df =  pd.concat([df_train, df_val])

    df_train.to_csv(join(output_folder, "train.csv"), index=False)
    df_val.to_csv(join(output_folder, "val.csv"), index=False)

    # custom split
    #################################################
    # patients = subdirs(join(root, "MR"), join=False)
    # task_name_variant1 = "Task037_CHAOS_Task_3_5_Variant1"
    # task_name_variant2 = 

    # output_preprocessed_v1 = join(preprocessing_output_dir, task_name_variant1)
    # maybe_mkdir_p(output_preprocessed_v1)

    # output_preprocessed_v2 = join(preprocessing_output_dir, task_name_variant2)
    # maybe_mkdir_p(output_preprocessed_v2)

    # splits = []
    # for fold in range(5):
    #     tr, val = get_split_deterministic(patients, fold, 5, 12345)
    #     train = ["T2_" + i for i in tr] + ["T1_" + i for i in tr]
    #     validation = ["T2_" + i for i in val] + ["T1_" + i for i in val]
    #     splits.append({
    #         'train': train,
    #         'val': validation
    #     })
    # save_pickle(splits, join(output_preprocessed_v1, "splits_final.pkl"))

    # splits = []
    # for fold in range(5):
    #     tr, val = get_split_deterministic(patients, fold, 5, 12345)
    #     train = ["T2_" + i for i in tr] + ["T1_in_" + i for i in tr] + ["T1_out_" + i for i in tr]
    #     validation = ["T2_" + i for i in val] + ["T1_in_" + i for i in val] + ["T1_out_" + i for i in val]
    #     splits.append({
    #         'train': train,
    #         'val': validation
    #     })
    # save_pickle(splits, join(output_preprocessed_v2, "splits_final.pkl"))

