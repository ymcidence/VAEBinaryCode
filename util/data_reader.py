import scipy.io as sio
import numpy as np


def reader_feat_only(i):
    total_batches = 1000
    data_folder = 'E:\\WorkSpace\\Data\\Hash\\Batch\\'
    file_name = data_folder + 'batch_' + str(i % total_batches + 1) + '.mat'
    mat_file = sio.loadmat(file_name)
    batch = dict()
    batch['batch_image'] = np.asarray(mat_file['batch_feat'], dtype=np.float32)
    batch['batch_label'] = np.asarray(mat_file['batch_label'], dtype=np.float32)
    return batch
