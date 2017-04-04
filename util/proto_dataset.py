import h5py
import numpy as np
from sklearn.utils import shuffle


class DatasetProto(object):
    def __init__(self, code_length, train_num, test_num, batch_size, file_path):
        self.code_length = code_length
        self.test_num = test_num
        self.train_num = train_num
        self.batch_size = batch_size
        self.file_path = file_path
        self.code_train = np.zeros((train_num, code_length))
        self.code_test = np.zeros((test_num, code_length))
        self._where_train = 0
        self._where_test = 0
        self.data_train, self.data_test, self.label_train, self.label_test = self._get_data()

    def _get_data(self):
        this_file = h5py.File(self.file_path, 'r')
        train_data = this_file['feat'][0:self.train_num]
        test_data = this_file['feat'][self.train_num: self.train_num + self.test_num]
        train_label = this_file['label'][0:self.train_num]
        test_label = this_file['label'][self.train_num: self.train_num + self.test_num]
        this_file.close()
        return train_data, test_data, train_label, test_label

    def next_batch_train(self):
        assert (self.train_num % self.batch_size) == 0
        ind_start = self._where_train
        ind_end = ind_start + self.batch_size
        batch_data = self.data_train[ind_start:ind_end]
        self._where_train = (self.batch_size + self._where_train)
        return batch_data

    def next_batch_test(self):
        assert (self.test_num % self.batch_size) == 0
        ind_start = self._where_test
        ind_end = ind_start + self.batch_size
        batch_data = self.data_test[ind_start:ind_end]
        self._where_test = (self.batch_size + self._where_test)
        return batch_data

    def iter_num(self):
        return int(self.train_num // self.batch_size), int(self.test_num // self.batch_size)

    def apply_code(self, codes, phase=0):
        if phase == 0:
            ind_start = self._where_train - self.batch_size
            ind_end = self._where_train
            self.code_train[ind_start:ind_end] = codes
            if self._where_train >= self.train_num:
                self._where_train = 0
        else:
            ind_start = self._where_test - self.batch_size
            ind_end = self._where_test
            self.code_test[ind_start:ind_end] = codes
            if self._where_test >= self.test_num:
                self._where_test = 0

    def reshuffle(self):
        self.data_train, self.label_train = shuffle(self.data_train, self.label_train)


class DatasetWithCluster(DatasetProto):
    def __init__(self, code_length, train_num, test_num, batch_size, file_path):
        super().__init__(code_length, train_num, test_num, batch_size, file_path)

        self.cluster_train, _  = self.

    def _get_cluster(self):
        this_file = h5py.File(self.file_path, 'r')
        train_cluster = this_file['cluster'][0:self.train_num]
        test_cluster = this_file['cluster'][self.train_num: self.train_num + self.test_num]
        return train_cluster, test_cluster
