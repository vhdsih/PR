#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from random import shuffle
from collections import Counter

# for 2 dims data


def draw_points(class_content):
    x = []
    y = []
    c = []
    color = 0
    for points in class_content:
        for point in points:
            x.append(point[0])
            y.append(point[1])
            c.append(color)
        color += 1
    plt.scatter(x, y, s=75, c=c, alpha=.5)
    plt.xlim(-5, 15)
    plt.xticks(())  # ignore xticks
    plt.ylim(-5, 15)
    plt.yticks(())  # ignore yticks
    plt.show()

# key function


def knn(data, n_class):
    # some variable
    counter = 0
    class_content = None
    class_centers = data[:n_class]
    vector_labels = np.empty((data.shape[0]), dtype=np.int32)

    while True:
        counter += 1
        print(' knn - times:', counter)
        class_content = [[] for i in range(n_class)]
        # for every vector, get a new class label by distance
        for i in range(data.shape[0]):
            vector = data[i]
            min_center = None
            min_distance = -1
            for j in range(len(class_centers)):
                distance = np.linalg.norm(vector - class_centers[j], ord=2)
                if min_distance == -1 or min_distance > distance:
                    min_distance = distance
                    min_center = j
            class_content[min_center].append(vector)
            vector_labels[i] = min_center

        # new centers of vectors
        new_class_centers = []
        for vectors in class_content:
            new_class_centers.append(np.mean(vectors, axis=0))

        # The boundary conditions
        new_class_centers = np.array(new_class_centers)
        if (new_class_centers == class_centers).all():
            break
        else:
            class_centers = new_class_centers.copy()

    # information
    print("Task done!")
    # draw_points(class_content)
    return vector_labels


if __name__ == '__main__':
    # test points - 2 dims
    # points = [
    #     [0, 0], [1, 0], [0, 1], [1, 1],
    #     [2, 1], [1, 2], [2, 2], [3, 2],
    #     [6, 6], [7, 6], [8, 6], [7, 7],
    #     [8, 7], [9, 7], [7, 8], [8, 8],
    #     [9, 8], [8, 9], [9, 9]
    # ]
    # data = np.array(points)

    # from csv
    data = np.loadtxt(open('ClusterSamples.csv'),
                             delimiter=",", skiprows=0).astype('float32') / 255
    labels = np.loadtxt(open('SampleLabels.csv'),
                              delimiter=None, skiprows=0).astype('int32')
    y = knn(data, 10)

    class_info = [[] for i in range(10)]
    for i in range(y.shape[0]):
        class_info[int(y[i])].append(labels[i])
    
    result = [Counter(x) for x in class_info]

    for r in result:
        print(r)