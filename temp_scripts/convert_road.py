import SimpleITK as sitk
import numpy as np
import nibabel as nib
from PIL import Image
import os


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

path = '/home/gandalf/nnunet/paths/road_segmentaion_ideal/training/input/img-2.png'
path_out = '/home/gandalf/segmento/data/Task120_MassRoadsSeg/img-2.nii.gz'
template = '/media/gandalf/data_storage/decatlon_brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz'
img = Image.open(path)
img_array = np.asarray(img)
img_array = np.rollaxis(img_array, 2)
img_array = np.expand_dims(img_array, 1)
sitk_img = sitk.GetImageFromArray(img_array, isVector=False)
temp_nib_img = nib.load(template)
sitk_img_temp = sitk.ReadImage(template)
numpyOrigin = np.array(list(reversed(sitk_img_temp.GetOrigin())))
numpySpacing = np.array(list(reversed(sitk_img_temp.GetSpacing())))
temp_sitk_array = sitk.GetArrayFromImage(sitk_img_temp)

img_array = np.rollaxis(np.rollaxis(np.rollaxis(img_array, 3), 3), 3)
shape_3d = img_array.shape[0:3]
rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
ras_pos = img_array.copy().view(dtype=rgb_dtype).reshape(shape_3d)  # copy used to force fresh internal structure
img_array = nib.Nifti1Image(img_array, np.eye(4))
nib.save(img_array, path_out)

#sitk_img.SetSpacing(numpySpacing)
#sitk_img.SetOrigin(numpySpacing)
#sitk.WriteImage(sitk_img, path_out)
#img = nib.Nifti1Image(img_array, np.eye(4))
#nib.save(img, path_out)

print('hejk')

