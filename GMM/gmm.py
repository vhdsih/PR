#!/usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=2, help='the mixture number')
args = parser.parse_args()

train_1 = 'data/Train1.csv'
train_2 = 'data/Train2.csv'
test_1 = 'data/Test1.csv'
test_2 = 'data/Test2.csv'

mnist_train_data = 'data/TrainSamples.csv'
mnist_train_label = 'data/TrainLabels.csv'

mnist_test_data = 'data/TestSamples.csv'
mnist_test_label = 'data/TestLabels.csv'


def read_data(data_path):
    return np.genfromtxt(data_path, delimiter=',')


def read_mnist_data(train=True):
    if train:
        data_path = mnist_train_data
        label_path = mnist_train_label
    else:
        data_path = mnist_test_data
        label_path = mnist_test_label

    data = read_data(data_path)
    label = read_data(label_path)

    if train:
        n_data = label.shape[0]
        splited_data = [list() for i in range(10)]
        for i in range(n_data):
            splited_data[int(label[i])].append(data[i])
        priors = []
        for i in range(10):
            priors.append(len(splited_data[i]) / n_data)
            splited_data[i] = np.array(splited_data[i])
        return splited_data, priors
    else:
        return data, label

# train1 = read_data(synthesis_train1_path)


class GMM:
    def __init__(self, K, eps=1e-6):
        self.K = K
        self.eps = eps

    def _gaussian(self, x, mean, cov):
        centered = x - mean
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        exponent = np.dot(np.dot(centered.T, cov_inv), centered)
        return np.exp(-0.5 * exponent) / np.sqrt(cov_det * np.power(2 * np.pi, len(mean)))

    def _init(self, data):
        data_shuffled = data.copy()
        np.random.shuffle(data_shuffled)
        data_splited = np.array_split(data_shuffled, self.K)

        self.means = np.array([np.mean(data_splited[i], axis=0)
                               for i in range(self.K)])
        self.covs = np.array([np.cov(data_splited[i].T)
                              for i in range(self.K)])
        self.alphas = np.repeat(1.0 / self.K, self.K)

    def fit(self, data):
        n_data = data.shape[0]

        norm_densities = np.empty((n_data, self.K), np.float)
        responsibilities = np.empty((n_data, self.K), np.float)

        old_log_likelihood = 0

        self._init(data)

        iteration = 0
        while True:
            for i in range(n_data):
                x = data[i]
                for j in range(self.K):
                    norm_densities[i][j] = self._gaussian(
                        x, self.means[j], self.covs[j])

            # log likelihood
            log_vector = np.log(
                np.array([np.dot(self.alphas, norm_densities[i]) for i in range(n_data)]))
            log_likelihood = log_vector.sum()

            if abs(old_log_likelihood - log_likelihood) < self.eps:
                break
            # E
            for i in range(n_data):
                x = data[i]
                normalizer = np.dot(self.alphas.T, norm_densities[i])
                for j in range(self.K):
                    responsibilities[i][j] = self.alphas[j] * \
                        norm_densities[i][j] / normalizer
            # M
            for i in range(self.K):
                responsibility = (responsibilities.T)[i]

                normalizer = np.dot(responsibility, np.ones(n_data))

                self.alphas[i] = normalizer / n_data
                self.means[i] = np.dot(responsibility, data) / normalizer
                diff = data - np.tile(self.means[i], (n_data, 1))
                self.covs[i] = np.dot((responsibility.reshape(
                    n_data, 1) * diff).T, diff) / normalizer

            old_log_likelihood = log_likelihood
            iteration += 1
            print('Iter : %d' % iteration)

    def get_result(self):
        print('alphas:', self.alphas)
        print('means:', self.means)
        print('covs:', self.covs)


class GMMClassifier:
    def __init__(self, gmms, priors):
        self.gmms = gmms
        self.priors = priors
        self.classes = len(priors)

    def classify(self, data, label):
        # data  n x d
        # label n
        n_data = data.shape[0]
        if isinstance(label, int):
            label = np.full((n_data,), label)
        log_vectors = np.empty((self.classes, n_data), dtype=np.float)
        for idx, gmm in enumerate(self.gmms):
            norm_densities = np.empty((n_data, gmm.K), np.float)
            for i in range(n_data):
                for j in range(gmm.K):
                    norm_densities[i][j] = gmm._gaussian(
                        data[i], gmm.means[j], gmm.covs[j])

            log_vectors[idx] = np.array(
                [np.dot(gmm.alphas, norm_densities[i]) for i in range(n_data)]) * self.priors[idx]

        predict = np.argmax(log_vectors, axis=0)
        n_correct = (predict == label).sum()
        accuracy = n_correct / n_data
        print('[%d/%d]=%.2f%%' % (n_correct, n_data, accuracy * 100))


if __name__ == '__main__':
    # gmm1 = GMM(2)
    # gmm1.fit(read_data(train_1))
    # gmm1.get_result()

    # gmm2 = GMM(2)
    # gmm2.fit(read_data(train_2))
    # gmm2.get_result()

    # classifier = GMMClassifier([gmm1, gmm2], [0.5, 0.5])
    # classifier.classify(read_data(test_1), 0)
    # classifier.classify(read_data(test_2), 1)

    mixture_num = 3
    print('K=%d' % mixture_num)
    gmms = [GMM(mixture_num) for i in range(10)]
    train_data, priors = read_mnist_data(train=True)
    # pdb.set_trace()
    for idx, (gmm, data) in enumerate(zip(gmms, train_data)):
        print('training %d' % idx)
        gmm.fit(data)

    mnist_classifier = GMMClassifier(gmms, priors)
    test_data, test_label = read_mnist_data(train=False)
    mnist_classifier.classify(test_data, test_label)