import scipy.io as sio
import numpy as np


def reader_feat_only(i, mode=0):
    total_batches = 145
    training_batches = 130
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
