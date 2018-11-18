import numpy as np

ratio = 0.9
normalize_paras_file = 'paras/normalize_para.npy'


class Data:
    def __init__(self, data_path, labels_path, train=True):

        self.train = train

        self.raw_data = np.loadtxt(open(data_path, 'r'), delimiter=",", skiprows=0).astype('float32')
        self.raw_labels = np.loadtxt(open(labels_path, 'r'), delimiter=None, skiprows=0).astype('int32')

        self.raw_data_size = int(self.raw_data.shape[0])
        self._normalize()
        self.global_index = 0

        self.train_size = int(ratio * self.raw_data.shape[0])
        self.test_size = self.raw_data_size - self.train_size
        self.train, self.test = self._split()

        if train:
            self.data = self.train
            self.data_size = self.train_size
        else:
            self.data = self.test
            self.data_size = self.test_size

    def _normalize(self):
        std = self.raw_data.std()
        mean = self.raw_data.mean()
        tmp = np.array([std, mean])
        np.save(normalize_paras_file, tmp)

        self.raw_data = (self.raw_data - mean) / std
        self._to_one_hot()

    def _to_one_hot(self, dimension=10):
        self.one_hot_labels = np.zeros((self.raw_data_size, dimension))
        for i, sequence in enumerate(self.raw_labels):
            self.one_hot_labels[i, sequence] = 1.

    def _split(self):
        train_data = self.raw_data[:self.train_size]
        train_labels = self.one_hot_labels[:self.train_size]
        test_data = self.raw_data[self.train_size:]
        test_labels = self.one_hot_labels[self.train_size:]
        return (train_data, train_labels), (test_data, test_labels)

    def next_batch(self, batch_size):
        if self.global_index + batch_size < self.data_size:
            batch_data = self.data[0][self.global_index:self.global_index + batch_size]
            batch_labels = self.data[1][self.global_index:self.global_index + batch_size]
        else:
            batch_data_0 = self.data[0][self.global_index:]
            batch_labels_0 = self.data[1][self.global_index:]
            batch_data_1 = self.data[0][:self.global_index + batch_size - self.data_size]
            batch_labels_1 = self.data[1][:self.global_index + batch_size - self.data_size]
            batch_data = np.concatenate((batch_data_0, batch_data_1), axis=0)
            batch_labels = np.concatenate((batch_labels_0, batch_labels_1), axis=0)
        self.global_index = (self.global_index + batch_size) % self.data_size
        return batch_data, batch_labels

    def get_test_data(self):
        return self.test
    
    def get_test_data_size(self):
        return self.test_size

    def get_data_size(self):
        return self.data_size


if __name__ == '__main__':
    d_p = 'data/data.csv'
    l_p = 'data/labels.csv'
    data = Data(d_p, l_p)
    print(data.get_data_size())
    data1 = Data(d_p, l_p, False)
    print(data1.get_data_size())
