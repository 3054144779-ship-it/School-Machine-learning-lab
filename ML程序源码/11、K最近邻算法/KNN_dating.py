# from __future__ import print_function
from numpy import *
import matplotlib.pyplot as plt
# import operator
# from os import listdir
# from collections import Counter

def file2matrix(file_name):
    """
    目的:导入训练数据
    params:file_name:数据文件路径
    returns:return_matrix:数据矩阵,class_label_vector:对应的类别
    """
    fr = open(file_name)
    str = fr.readlines()    # 读取文件内容(这会清空缓存区中的数据)
    number_lines = len(str)  # 获得文件中的数据行的行数
    return_matrix = zeros((number_lines,3)) # 准备返回矩阵;生成对应的空矩阵,eg:zeros(2,3):生成一个2*3的矩阵,各个位置上全0
    class_label_vector = [] # 准备返回标签
    index = 0
    for line in str:
        line_valid = line.strip()   # 删除字符串两端的空白字符
        list_from_line_valid = line_valid.split('\t')   # 以'\t'切割字符串
        return_matrix[index,:] = list_from_line_valid[0:3]
        class_label_vector.append(int(list_from_line_valid[-1]))    # 每行的类别数据,就是label标签数据,访问列表list_from_line_valid的最后一个元素
        index += 1
    fr.close()
    return return_matrix,class_label_vector # 返回数据矩阵return_matrix和对应的类别class_label_vector

def plot_scatter(matrix_data, matrix_labels):
    figure = plt.figure()
    sp = figure.add_subplot(111)
    sp.scatter(matrix_data[:, 0], matrix_data[:, 1], 15.0 * array(matrix_labels), 15.0 * array(matrix_labels))
    plt.show()

def auto_norm(data_set):
    """
    目的:归一化特征值,消除属性之间量级不同导致的影响
    params:data_set:数据集
    returns:norm_data_set:归一化后的数据集,ranges:范围,min_values:最小值
    归一化公式:
    Y=(X-Xmin)/(Xmax-Xmin),其中的min和max分别是数据集中的最小特征值和最大特征值,该函数可以自动将数字特征值转化为0到1的区间
    """
    # 计算每种属性的最小值,最大值,范围
    min_values = data_set.min(0)    # 计算数据集矩阵data_set中每列(0)的最小值
    max_values = data_set.max(0)    # 计算数据集矩阵data_set中每列(0)的最大值
    ranges = max_values - min_values    # 计算数据集矩阵data_set中每列的极差
    # norm_data_set = zeros(shape(data_set))  # 准备返回的归一化后的数据集矩阵(形状与数据集data_set相同);生成对应的空矩阵
    m = data_set.shape[0]   # m:数据集data_set的第一个维度(即行数)
    norm_data_set = data_set - tile(min_values,(m,1))   # tile:构造m*1型的矩阵,矩阵中的各元素为min_values;生成与最小值之差组成的矩阵
    norm_data_set = norm_data_set / tile(ranges,(m,1))  # 将上述矩阵除以范围组成的m*3矩阵(各个元素分别相除)
    return  norm_data_set,ranges,min_values

def classify_knn(x_in,data_set,labels,k):
    """
    params:
        x_in:要分类的输入向量
        data_set:输入的训练样本集
        labels:标签向量(输入的训练样本集对应的标签向量)
        k:选择最近邻的数目
    notes:
        labels元素数目=data_set行数
        程序使用欧拉距离公式
    """
    # 1.计算距离
    data_set_size = data_set.shape[0]   # data_set_size:数据集data_set的第一个维度(即行数)
    diff_matrix = tile(x_in,(data_set_size,1))-data_set  # 使用tile生成和训练样本对应的矩阵,并与训练样本求差
    """
    欧氏距离:点到点之间的距离
    """
    square_diff_matrix = diff_matrix ** 2   # 每个元素取平方构成矩阵square_diff_matrix
    square_distance_vector = square_diff_matrix.sum(axis=1)   # 将矩阵square_diff_matrix的每一行中所含的各个元素相加,构成一个square_diff_matrix.shape[0](亦即data_set_size)*1的向量square_distance_vector
    distances_vector = square_distance_vector ** 0.5    # 将向量square_distance_vector中的每个元素开方后得到向量distances_vector
    """
    y=x.argsort():将x中的元素升序排列,提取其对应的索引,然后输出到y
    eg:x=array([1,4,3,-1,6,9]),y=x.argsort()=array([3,0,2,1,4,5]);x[3]=-1(min),y[0]=3;x[5]=9(max),y[5]=5
    """
    index_ascend_sorted_vector = distances_vector.argsort() # 根据距离从小到大排序,返回对应的索引,构成相应的向量index_ascend_sorted_vector
    # 2.选择距离最小的k个点
    class_count = {}
    for i in range(k):
        # print(index_ascend_sorted_vector[i])
        vote_label = labels[index_ascend_sorted_vector[i]]  # 找到该样本的类型
        """
        字典的get方法:
        dict.get(k,d),其中:get相当于一条if...else...语句;参数k在字典中,字典将返回dict[k](即k对应的value值);如果参数k不在字典中,则返回参数d
        """
        class_count[vote_label] = class_count.get(vote_label, 0) + 1    # 在字典中将该类型对应的计数量+1
    # 3.排序并返回出现次数最多的那个类型
    key_max_class_cnt = max(class_count, key=class_count.get)   # 利用函数max直接返回字典中value最大的键
    return key_max_class_cnt

def dating_class_test():
    """
    目的:对约会网站的测试方法
    return:无
    """
    # 设置测试数据的一个比例(训练数据集比例=1-ho_ratio)
    ho_ratio = 0.1  # 测试范围,一部分测试,一部分作为样本
    # 从文件中加载数据
    dating_data_matrix, dating_labels = file2matrix('./2.KNN/datingTestSet2.txt')
    plot_scatter(dating_data_matrix, dating_labels)
    # print(dating_data_matrix)
    # print(dating_labels)
    # 归一化数据
    norm_matrix, ranges, min_values = auto_norm(dating_data_matrix)
    # m:数据集的行数,即矩阵的第一维
    m = norm_matrix.shape[0]
    # 设置测试的样本数量,number_test_vectors表示测试样本的数量
    number_test_counts = int(m * ho_ratio)
    print('number_test_counts=',number_test_counts)
    error_counts = 0
    for i in range(number_test_counts):
        # 对数据测试
        """
        测试集:norm_matrix[0:number_test_counts, :]
        训练集:norm_matrix[number_test_counts:m, :]
        标签集:dating_labels[number_test_counts:m]
        最近邻数目:3
        """
        classify_result = classify_knn(norm_matrix[i, :], norm_matrix[number_test_counts:m, :], dating_labels[number_test_counts:m], 3)
        print("分类结果:%d,真正结果:%d" % (classify_result, dating_labels[i]))
        if classify_result != dating_labels[i]:
            error_counts += 1
    print(error_counts)
    print("总错误率:%f" % (error_counts/float(number_test_counts)))

if __name__ == '__main__':
    dating_class_test()