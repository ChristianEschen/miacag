import pickle
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd


def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def saveImgs(img, c_batch, c_img, path):
    array = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path, c_batch, c_img) + '.jpg', array)
    path_out_img = os.path.join(c_batch, c_img) + '.jpg'
    return path_out_img


def getbatch(liste, outpath, labels_list, path_list):
    c_batch = 0
    for path in liste:
        mkDir(os.path.join(outpath, str(c_batch)))
        pickle_dict = unpickle(path)
        label_list, path_list = getImgs(pickle_dict, c_batch,
                                        outpath, labels_list, path_list)
        c_batch += 1
    return label_list, path_list


def getImgs(pickle_dict, c_batch, path, labels_list, path_list):
    c_img = 1
    images = np.reshape(pickle_dict['data'], (10000, 3, 32, 32))
    for i in range(0, images.shape[0]):
        img = images[i]
        img = np.transpose(img, (1, 2, 0))
        path_img = saveImgs(img, str(c_batch), str(c_img), path)
        label = pickle_dict['labels'][i]
        c_img += 1
        labels_list.append(label)
        path_list.append(path_img)
    return labels_list, path_list


def main():
    # Train
    labels_list = []
    path_list = []
    outpath = '/media/gandalf/AE3416073415D2E7/cifar10/cifar10_rgb/'
    paths = ["/media/gandalf/AE3416073415D2E7/cifar10/cifar-10-batches-py/data_batch_1",
            "/media/gandalf/AE3416073415D2E7/cifar10/cifar-10-batches-py/data_batch_2",
            "/media/gandalf/AE3416073415D2E7/cifar10/cifar-10-batches-py/data_batch_3",
            "/media/gandalf/AE3416073415D2E7/cifar10/cifar-10-batches-py/data_batch_4",
            "/media/gandalf/AE3416073415D2E7/cifar10/cifar-10-batches-py/data_batch_5"]
    labels_list, path_list = getbatch(paths, outpath, labels_list, path_list)
    df_train = pd.DataFrame(
        list(zip(path_list, labels_list)),
        columns=['inputs', 'labels'])
    #df_train['labels'] = df_train['labels'].sample(frac=0.1)
    df_train.to_csv(os.path.join(outpath, 'train_full_labels.csv'))
    
    # test
    labels_list = []
    path_list = []
    outpath = '/media/gandalf/AE3416073415D2E7/cifar10/cifar10_rgb_test/'
    mkDir(outpath)
    paths = ["/media/gandalf/AE3416073415D2E7/cifar10/cifar-10-batches-py/test_batch"]
    labels_list, path_list = getbatch(paths, outpath, labels_list, path_list)
    df_test = pd.DataFrame(
        list(zip(path_list, labels_list)),
        columns=['inputs', 'labels'])
    df_test.to_csv(os.path.join(outpath, 'test.csv'))

if __name__ == '__main__':
    main()
