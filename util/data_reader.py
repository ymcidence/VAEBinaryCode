import scipy.io as sio
import numpy as np


def reader_feat_only(i, mode=0):
    total_batches = 1030
    training_batches = 1015
    test_batches = total_batches - training_batches
    data_folder = 'E:\\WorkSpace\\Data\\sun397\\Batch2\\'
    if mode == 0:
        file_name = data_folder + 'batch_' + str(i % training_batches + 1) + '.mat'
    else:
        file_name = data_folder + 'batch_' + str(i % test_batches + 1 + training_batches) + '.mat'
    mat_file = sio.loadmat(file_name)
    batch = dict()
    batch['batch_feat'] = np.asarray(mat_file['batch_feat'], dtype=np.float32)
    batch['batch_label'] = np.asarray(mat_file['batch_label'], dtype=np.float32)
    return batch


def reader_cifar10(i, mode=0):
    total_batches = 600
    training_batches = 590
    test_batches = total_batches - training_batches
    data_folder = 'E:\\WorkSpace\\Data\\CIFAR10\\Batch\\'
    if mode == 0:
        file_name = data_folder + 'batch_' + str(i % training_batches + 1) + '.mat'
    else:
        file_name = data_folder + 'batch_' + str(i % test_batches + 1 + training_batches) + '.mat'
    mat_file = sio.loadmat(file_name)
    batch = dict()
    batch['batch_feat'] = np.asarray(mat_file['batch_feat'], dtype=np.float32)
    batch['batch_label'] = np.asarray(mat_file['batch_label'], dtype=np.float32)
    return batch


def code_label_saver(codes, labels):
    sio.savemat('E:\\WorkSpace\\WorkSpace\\ITQ\\hehe3.mat', {'codes': codes, 'labels': labels})
