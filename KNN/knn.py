#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from random import shuffle

def draw_points(class_content):
    x = []
    y = []
    c = []
    color = 0
    for  points in class_content:
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

def knn(input_data, n_class):
    # shuffle data
    np.random.shuffle(input_data)
    # some variable
    counter = 0
    class_content = None
    class_centers = input_data[:n_class]

    while True:
        counter += 1
        print(' ', counter)
        class_content = [[] for i in range(n_class)]
        # for every vector, get a new class label by distance
        for vector in input_data:
            min_center = None
            min_distance = -1
            for i in range(len(class_centers)):
                distance = np.linalg.norm(vector - class_centers[i]) 
                if min_distance == -1 or min_distance > distance:
                    min_distance = distance
                    min_center = i
            class_content[min_center].append(vector)
            
        # get new centers of vectors
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
    # print(class_content)
    # draw_points(class_content)
        


if __name__ == '__main__':
    # test points
    points = [[0, 0], [1, 0], [0, 1], [1, 1],
    [2, 1], [1, 2], [2, 2], [3, 2],
    [6, 6], [7, 6], [8, 6], [7, 7],
    [8, 7], [9, 7], [7, 8], [8, 8],
    [9, 8], [8, 9], [9, 9]]
    input_data = np.array(points)
    # from csv
    input_data = np.loadtxt(open('ClusterSamples.csv'), delimiter=",", skiprows=0)
    knn(input_data, 10)
    # 10000 784 10
    # para1: 2 dimension char * pixels; para2: n_class
    # distance function: numpy.linalg.norm(v1 - v2)
    # 

