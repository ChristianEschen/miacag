import SimpleITK as sitk
import numpy as np
import nibabel as nib
# nifty_brats = '/home/gandalf/segmento/data/MICCAI_BraTS2020_TrainingData/image_data_raw_subset_same_data/images/BraTS20_Training_002/BraTS20_Training_002_0002.nii.gz'
# brats_img = sitk.ReadImage(nifty_brats)
# array = sitk.GetArrayFromImage(brats_img)
# nib_array = nib.load(nifty_brats)

dcm_ang = '/home/gandalf/Documents/dcms/angs/dcm/2/0002.DCM'
nifty_ang = '/home/gandalf/segmento/data/0002.nii.gz'
nifty_sitk_ang = '/home/gandalf/segmento/data/0002sitk.nii.gz'
nrrd_ang = '/home/gandalf/segmento/data/0002.nrrd'
img = sitk.ReadImage(dcm_ang)
array = sitk.GetArrayFromImage(img)
img = nib.Nifti1Image(array, np.eye(4).astype('uint8'))
img.header.set_data_dtype(np.uint8)
nib.save(img, nifty_ang)
sitk.WriteImage(sitk.GetImageFromArray(array), nifty_sitk_ang)
sitk.WriteImage(sitk.GetImageFromArray(array), nrrd_ang)
# array_sitk = sitk.GetArrayFromImage(img)
# #sitk.WriteImage(img, nifty_ang)
# img_write = sitk.ReadImage(nifty_ang)
# ang_nii_array = sitk.GetArrayFromImage(img_write)
# data = np.transpose(np.expand_dims(array_sitk,-1), (1, 2, 3, 0))
# img = nib.Nifti1Image(data, np.eye(4))
# #hdr = img.header
# nib.save(img, nifty_ang)
print('hej')