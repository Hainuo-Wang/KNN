import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Generate_Random_Central_Dots(dataset, k):
    '''
    :param dataset: shape:(m, n)  m个点，每个点有n个维度
    :param k: 分成k类，找k个中心点
    :return: central_dots, (k, n)
    '''
    features_dim = dataset.shape[1]  # n, feature number of each dot
    central_dots = np.zeros(shape=(k, features_dim))
    for i in range(features_dim):
        dim_i_min = min(dataset[:, i])  # m个点，在维度i上的最小值
        dim_i_max = max(dataset[:, i])  # m个点，在维度i上的最大值
        dim_i_range = dim_i_max - dim_i_min
        central_dots[:, i] = (dim_i_min + dim_i_range * np.random.rand(k, 1)).reshape(central_dots[:, i].shape)
    return central_dots

def Calculate_Distance(dot_1, dot_2):
    '''
    :param dot_1: (n,)
    :param dot_2: (n,)
    :return: 欧氏距离
    '''
    return np.sqrt(sum(np.power(dot_1 - dot_2, 2)))  # np.power(x, 2):x**2

def KNN(dataset, k):
    '''
    :param dataset: 散点集,(m, n)
    :param k: 分类个数
    :return: central_dots_array: k个中心点,(k,n);type_distance_array:m个点的类型以及于中心点的欧氏距离,(m,2)
    '''
    dataset = np.array(dataset)  # 根据类型自行删除
    num_dots = dataset.shape[0]
    type_distance_array = np.zeros(shape=(num_dots, 2))  # dim_0:type, dim_1:distance
    central_dots_array = Generate_Random_Central_Dots(dataset, k)
    flag = True  # 是否已经收敛（新一轮中心点不改变）
    while flag:
        flag = False
        for i in range(num_dots):  # 遍历每一个散点,进行分类,并求与最近中心点的欧氏距离
            min_index = -1
            min_distance = np.inf  # +∞,是没有确切的数值的,类型为浮点型
            for j in range(k):  # 找到与散点i的欧式距离最近的中心点j，将其索引保存到min_index,最短距离保存到min_distance
                distance = Calculate_Distance(dataset[i], central_dots_array[j])
                if distance < min_distance:
                    min_distance = distance
                    min_index = j
            if type_distance_array[i][0] != min_index:
                type_distance_array[i] = min_index, min_distance
                flag = True
        for j in range(k):  # 聚类之后，更新中心点
            li = []  # 一个聚类内的所有点
            for i in range(num_dots):
                if type_distance_array[i][0] == j:
                    li.append(dataset[i])
            central_dots_array[j, :] = np.mean(li, axis=0)  # 聚类内每个点各维度的平均值
    return central_dots_array, type_distance_array
