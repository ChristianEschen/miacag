import SimpleITK as sitk
import nibabel as nib
import os

mri_path = '/media/gandalf/AE3416073415D2E7/CHAOS_pancreas_MRI/Train_Sets/MR/'
output_train = '/media/gandalf/AE3416073415D2E7/CHAOS_pancreas_MRI_nifty/train/'
output_val = '/media/gandalf/AE3416073415D2E7/CHAOS_pancreas_MRI_nifty/val/'


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def getImagesPaths(path):
    images_T1inPhase_path = []
    images_T1outPhase_path = []
    images_T2_path = []
    labels_path = []
    dirs = os.listdir(path)
    for subject in dirs:
        subject_path = os.path.join(path, subject)
        modalities = os.listdir(subject_path)
        for modality in modalities:
            mod_path = os.path.join(subject_path, modality)
            image_type = os.listdir(mod_path)
            for img in image_type:
                if img == 'DICOM_anon':
                    image = os.path.join(mod_path, img)
                    sub_modal_images = os.listdir(image)
                    for sub_modal in sub_modal_images:
                        image_sub = os.path.join(image, sub_modal)
                        if sub_modal == 'inPhase':
                            images_T1inPhase_path 
                        elif sub_modal == 'inPhase':
                            print('j')
                        else:
                            ValueError('not correct sub modal')
                        images_T1inPhase_path.append((image_sub, subject))
                elif img == 'Ground':
                    labels = os.path.join(mod_path, img)
                    labels_path.append((labels, subject))
                else:
                    ValueError('unknown folder')
    return images_path, labels_path


def load_and_write_Images(image_paths, label_paths, out_train, out_val):
    image_path_train = image_paths[0:int(0.7*len(image_paths))]
    image_path_val = image_paths[int(0.7*len(image_paths)):]
    for idx in range(0, len(image_path_train)):
        image_path = image_paths[idx]
        label_path = label_paths[idx]
        os.system("dcm2niix " + "-o" + os.path.join(out_train, image_path[1]+ '.nii.gz') + image_path[0])
        print('hje')
        #image, label = getImagePairs(image_path, label_path)
    return None

def getImagePairs(image_path, label_path):
    return image, label


def main():
    mkDir(output_train)
    mkDir(output_val)
    images_paths, labels_paths = getImagesPaths(mri_path)
    load_and_write_Images(images_paths, labels_paths, output_train, output_val)


if __name__ == '__main__':
    main()
