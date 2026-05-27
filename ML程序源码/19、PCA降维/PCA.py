from math import isnan
from numpy import *
import matplotlib.pyplot as plt

def load_data_set(name_file, separator = '\t'):
    fr = open(name_file)
    str_all = fr.readlines()
    fr.close()
    array_str = [str_line.strip().split(separator) for str_line in str_all]
    array_data = [list(map(float, array_line)) for array_line in array_str] # 这里和python2的区别,需要在map函数外加一个list()
    return mat(array_data)

def PCA(matrix_data, top_N_feature = 9999999):
    """
    args:matrix_data:原数据集矩阵;top_N_feature:应用的N个特征
    returns:matrix_dimension_reduce:降维后数据集;matrix_data_new_space:新的数据集空间
    """
    values_mean = mean(matrix_data, axis = 0)   # 计算每一列的均值
    matrix_data_mean_removed = matrix_data - values_mean    # 每个向量同时都减去均值
    """
    cov(协方差) = [(x1 - x均值) * (y1 - y均值) + (x2 - x均值) * (y2 - y均值) + ... + (xn - x均值) * (yn - y均值)] / (n - 1)
    方差:(一维)度量两个随机变量关系的统计量
    协方差:(二维)度量各个维度偏离其均值的程度
    协方差矩阵:(多维)度量各个维度偏离其均值的程度
    当cov(X, Y) > 0时,表明X与Y正相关(X越大,Y也越大;X越小,Y也越小;这种情况,称为"正相关")
    当cov(X, Y) < 0时,表明X与Y负相关
    当cov(X, Y) = 0时,表明X与Y不相关
    """
    matrix_cov = cov(matrix_data_mean_removed, rowvar = 0)
    value_eigen, vector_value_eigen = linalg.eig(mat(matrix_cov))   # value_eigen为特征值,vector_value_eigen为特征向量
    """
    对特征值,进行从小到大的排序,返回从小到大的索引序号
    特征值的逆序就可以得到top_N_feature个最大的特征向量
    """
    index_value_eigen = argsort(value_eigen)
    index_value_eigen = index_value_eigen[:(-(top_N_feature + 1)):(-1)] # -1表示倒序,返回top_N_feature的特征值[-1到-(top_N_feature + 1),但是不包括-(top_N_feature + 1)本身的倒序]
    vector_eigen_red = vector_value_eigen[:, index_value_eigen] # 重组vector_value_eigen(最大到最小)
    """
    将数据转换到新空间
    """
    matrix_dimension_reduce = matrix_data_mean_removed * vector_eigen_red
    matrix_data_new_space = (matrix_dimension_reduce * vector_eigen_red.T) + values_mean
    return matrix_dimension_reduce, matrix_data_new_space

def replace_NaN_with_mean():
    matrix_data = load_data_set('./13.PCA/secom.data', ' ')
    number_feature = shape(matrix_data)[1]
    for i in range(number_feature):
        value_mean = mean(matrix_data[nonzero(~isnan(matrix_data[:, i].A))[0], i])  # 对值不为NaN的求均值;.A返回矩阵基于的数组
        matrix_data[nonzero(isnan(matrix_data[:, i].A))[0], i] = value_mean # 将值为NaN的值赋值为均值
    return matrix_data

def show_picture(matrix_data, matrix_data_new_space):
    figure = plt.figure()
    sp = figure.add_subplot(111)
    sp.scatter(matrix_data[:, 0].flatten().A[0], matrix_data[:, 1].flatten().A[0], marker = '^', s = 90)
    sp.scatter(matrix_data_new_space[:, 0].flatten().A[0], matrix_data_new_space[:, 1].flatten().A[0], marker = 'o', s = 50, c = 'red')
    plt.show()

def analyse_data(matrix_data):
    values_mean = mean(matrix_data, axis = 0)
    matrix_data_mean_removed = matrix_data - values_mean
    matrix_cov = cov(matrix_data_mean_removed, rowvar = 0)
    value_eigen, vector_value_eigen = linalg.eig(mat(matrix_cov))
    index_value_eigen = argsort(value_eigen)
    top_N_features = 20
    index_value_eigen = index_value_eigen[:(-(top_N_features + 1)):(-1)]
    cov_all_score = float(sum(value_eigen))
    sum_cov_score = 0
    for i in range(0, len(index_value_eigen)):
        line_cov_score = float(value_eigen[index_value_eigen[i]])
        sum_cov_score += line_cov_score
        """
        会发现其中有超过20%的特征值都是0
        这就意味着这些特征都是其他特征的副本,也就是说,它们可以通过其他特征来表示,而本身并没有提供额外的信息
        最前面15个值的数量级大于10^5,实际上那以后的值都变得非常小
        这就相当于告诉我们只有部分重要特征,重要特征的数目也很快就会下降
        最后,可能会注意到有一些小的负值,他们主要源自数值误差应该四舍五入成0
        """
        print('主成分:%s,方差占比:%s%%,累积方差占比:%s%%' % (format(i + 1, '2.0f'), format(line_cov_score / cov_all_score * 100, '4.2f'), format(sum_cov_score / cov_all_score * 100, '4.1f')))

if __name__ == '__main__':
    matrix_data = replace_NaN_with_mean()   # 利用PCA对半导体制造数据降维
    print(shape(matrix_data))
    analyse_data(matrix_data)   # 分析数据
