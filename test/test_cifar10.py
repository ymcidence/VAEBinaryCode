import tensorflow as tf
from model.hehe import ConditionalVairationalBinaries as Model
from util.proto_dataset import DatasetWithCluster

PATH_SUMMARY = 'E:\\WorkSpace\\WorkSpace\\TrainingLogs\\BVAE\\CIFAR10_16'
PATH_SNAPSHOT = 'E:\\WorkSpace\\WorkSpace\\SavedModels\\BVAE\\CIFAR10_16'
PATH_DATA = 'E:\\WorkSpace\\Data\\CIFAR10\\cifar_data.hdf5'

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    this_session = tf.Session(config=config)

    dataset = DatasetWithCluster(16, 59000, 1000, 200, PATH_DATA)

    model_config = {
        'sess': this_session,
        'batch_size': 200,
        'feature_length': 4096,
        'code_length': 16,
        'latent_size': 1024,
        'label_num': 10,
        'restore_file': None
    }

    model = Model(**model_config)

    model.train(dataset, PATH_SUMMARY, PATH_SNAPSHOT)
