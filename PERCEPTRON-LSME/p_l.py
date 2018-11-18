import numpy as np


class Perceptron:
    def __init__(self, data, labels):
        self.raw_data = data
        self.raw_labels = labels
        self.data_size, self.n_dim = data.shape

        self.data = None
        self.weight = None

        self._init_dataset()
        self._init_perceptron()

    def _init_dataset(self):
        tmp = np.ones((self.data_size, 1))
        self.data = np.concatenate((self.raw_data, tmp), axis=1)
        for i in range(self.data_size):
            self.data[i] *= self.raw_labels[i]

    def _init_perceptron(self):
        self.weight = np.random.randn(self.n_dim + 1)

    def update(self):
        yes = False
        for i in range(self.data_size):
            if np.dot(self.weight, self.data[i]) < 0:
                self.weight += self.data[i]
                yes = True
        return yes

    def test(self, data, labels):
        tmp = np.ones((data.shape[0], 1))
        data = np.concatenate((data, tmp), axis=1)

        for x in data:
            print(np.dot(self.weight, x))

    def get_weight(self):
        return self.weight


class KeslerPerceptron:
    def __init__(self, data, labels, n_class, lr=0.0000001):
        self.data = data
        self.labels = labels
        self.n_class = n_class

        self.lr = lr

        self.n_dim = self.data.shape[1]
        self.data_size = self.data.shape[0]
        self._init_perceptron()

    def _init_perceptron(self):
        self.weight = np.random.randn(
            self.n_class*self.n_dim).reshape((
                self.n_class, self.n_dim)).astype('float32')

    def _calculate(self, i, data):
        values = []
        for j in range(self.n_class):
            value = np.dot(self.weight[j], data[i].T)
            values.append(value)
        return values

    def update(self):
        for i in range(self.data_size):
            y = self.labels[i]
            values = self._calculate(i, self.data)
            for j in range(self.n_class):
                if j != y and values[y] - values[j] <= 0:
                    self.weight[y] += self.lr * self.data[i]
                    self.weight[j] -= self.lr * self.data[i]

    def test(self, test_data, test_labels):
        n = 0
        for i in range(test_data.shape[0]):
            values = self._calculate(i, test_data)
            r = np.argmax(values)
            if r == test_labels[i]:
                n += 1
        return n / test_data.shape[0]

    def save(self):
        np.save('k_p_weight.npy', self.weight)


class LMSE:
    def __init__(self, data, labels):
        self.raw_data = data
        self.raw_labels = labels
        self.data_size, self.n_dim = data.shape

        self._init_dataset()
        self._init_lmse()

    def _init_dataset(self):
        tmp = np.ones((self.data_size, 1))
        self.data = np.concatenate((self.raw_data, tmp), axis=1)
        for i in range(self.data_size):
            self.data[i] *= self.raw_labels[i]

    def _init_lmse(self):
        self.b = np.ones(self.data_size)
        self.X = np.linalg.inv(np.dot(self.data.T, self.data)).dot(self.data.T)
        self.w = np.dot(self.X, self.b)
        print(self.w.shape)

    def predict(self):
        tmp = np.ones((self.data_size, 1))
        self.test = np.concatenate((self.raw_data, tmp), axis=1)
        for i in range(self.data_size):
            v = np.dot(self.w, self.test[i])
            if v > 0:
                print(self.test[i], 0)
            else:
                print(self.test[i], 1)

    def get_weight(self):
        return self.w


def load_data(d_p, l_p):
    data = np.loadtxt(open(d_p, 'r'), delimiter=",",
                      skiprows=0).astype('float32')
    labels = np.loadtxt(open(l_p, 'r'), delimiter=None,
                        skiprows=0).astype('int32')
    return data, labels


if __name__ == '__main__':
    '''
    exp1: kesler perceptron for multi-classes classification
    '''
    # kesler perceptron: multi-classes
    train_data_path = 'TrainSamples.csv'
    train_labels_path = 'TrainLabels.csv'
    test_data_path = 'TestSamples.csv'
    test_labels_path = 'TestLabels.csv'

    train_data, train_labels = load_data(train_data_path, train_labels_path)
    test_data, test_labels = load_data(test_data_path, test_labels_path)

    n_class = 10
    perceptron = KeslerPerceptron(train_data, train_labels, n_class)
    for epoch in range(200):
        perceptron.update()
        print(epoch, perceptron.test(test_data, test_labels))
    perceptron.save()
    '''
    2: perceptron algorithm for 2-classes
    '''

    # data = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]])
    # labels = np.array([1, 1, 1, -1, -1, -1])
    # perceptron = Perceptron(data, labels)

    # start = True
    # while start:
    #     start = perceptron.update()
    # print(perceptron.get_weight())
    # perceptron.test(data, labels)

    '''
    3: lsme
    '''
    # lmse = LMSE(data, labels)
    # print(lmse.get_weight())
    # lmse.predict()
