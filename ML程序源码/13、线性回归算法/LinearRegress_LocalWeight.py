from numpy import *
import matplotlib.pylab as plt
from numpy import asmatrix as mat

def load_data_set(file_name):
    """
    desc:加载数据,解析以tab键分隔的文件中的浮点数
    args:file_name:文件路径名
    returns:data_matrix:特征对应的数据集;label_vector:特征对应的分类标签,即类别标签
    """
    fr = open(file_name)
    str_all = fr.readlines()
    fr.close()
    data_array = []
    label_array = []
    for str_line in str_all:
        cur_line = str_line.strip().split('\t') # 1.删掉字符串str_line中最左边和最右边的空白字符得到相应的字符串2.以tab分割上述字符串得到字符串列表cur_line
        line_array = []
        for i in range(len(cur_line) - 1):   # 除去字符串列表cur_line中的最后一列外
            line_array.append(float(cur_line[i]))   # 将数据添加到列表line_array中,每行的训练数据构成一个行向量
        data_array.append(line_array)  # 将训练数据的输入数据部分存储到列表data_matrix中
        label_array.append(float(cur_line[-1]))    # 将每行的最后一个数据,即类别,或叫目标变量存储到列表label_vector中
    """
    函数mat是将array转换为矩阵的函数
    """
    data_matrix = mat(data_array)   # data_matrix(形状:(m,2))
    label_vector = mat(label_array) # label_vector(形状:(1,m))!!!
    # print(data_matrix)
    # print(label_vector)
    return data_matrix, label_vector

def lwlr(test_point, x_matrix, y_vector, k = 1.0):
    """
    desc:局部加权线性回归,在待预测点附近的每个点赋予一定的权重,在子集上基于最小均方差进行普通的回归
    args:test_point:样本点;x_matrix:样本的特征数据;y_vector:每个样本对应的类别标签,即目标变量;k:关于赋予权重矩阵的核的一个参数,与权重的衰减速率有关
    returns:test_point * ws:数据点与具有权重的系数相乘得到的预测点
    notes:
        这其中会用到计算权重的公式:w=e^((x^(i)-x)/(-2*k^2)
        理解:
            x为某个预测点,x^(i)为样本点,样本点距离预测点越近,贡献的误差越大(权值越大),越远则贡献的误差越小(权值越小)
            关于预测点的选取,在本代码中取的是样本点;其中k是带宽参数,控制w(钟形函数)的宽窄程度,类似于高斯函数的标准差
        算法思路:
                假设预测点取样本点中的第i个样本点(共m个样本点),遍历1到m个样本点(含第i个),算出每一个样本点与预测点的距离;也就可以计算出每个样本贡献误差的权值,可以看出w是一个有m个元素的向量(写成对角阵形式)   
    """
    y_matrix = y_vector.T   # mat().T是转换为矩阵之后,再进行转置操作;y_matrix(shape:(m,1))
    m = shape(x_matrix)[0]  # 获得矩阵x_matrix的行数
    weights = mat(eye(m))   # eye()返回一个对角线元素为1,其它元素为0的二维数组;创建权重矩阵weights,该矩阵为每个样本点初始化了一个权重
    for i in range(m):
        difference_matrix = test_point - x_matrix[i]    # test_point的形式是一个行向量的形式;计算test_point与输入样本点之间的距离
        weights[i, i] = exp((difference_matrix * difference_matrix.T)/(-2.0*k**2))  # 计算出每个样本贡献误差的权值;k:控制衰减的速率
    xTx = x_matrix.T * (weights * x_matrix) # 根据矩阵乘法计算xTx,其中的矩阵weights是样本点对应的权重矩阵
    if linalg.det(xTx)  ==  0.0:
        print("该矩阵为奇异矩阵,不可逆")
        return []
    ws = xTx.I * (x_matrix.T * (weights * y_matrix))    # 计算出回归系数的一个估计
    return test_point * ws

def lwlr_test(test_matrix, x_matrix, y_vector, k=1.0):
    """
    desc:测试局部加权线性回归,对数据集中每个点调用函数lwlr()
    args:test_matrix:测试所用的所有样本点;x_matrix:样本的特征数据;y_vector:每个样本对应的类别标签,即目标变量;k:控制核函数的衰减速率
    returns:y_hat:预测点的估计值
    """
    m = shape(test_matrix)[0]   # 得到样本点的总数
    y_hat = zeros(m)    # 构建一个全都是0的1*m矩阵
    """
    循环所有的数据点,并将lwlr运用于所有的数据点
    """
    for i in range(m):
        y_hat[i] = lwlr(test_matrix[i], x_matrix, y_vector, k)
    return y_hat    #返回估计值

def line_regress_local_weight():
    x_matrix, y_vector = load_data_set('./8.Regression/data.txt')
    y_hat = lwlr_test(x_matrix, x_matrix, y_vector, 0.003)
    sort_index = x_matrix[:, 1].argsort(0)  # 函数argsort()是将x中的元素从小到大排列,提取其对应的索引index,然后输出
    x_sort_matrix = x_matrix[sort_index][:, 0, :] # :,0,:是什么意思?
    figure = plt.figure()
    sp = figure.add_subplot(111)
    sp.plot(x_sort_matrix[:, 1], y_hat[sort_index])
    sp.scatter(x_matrix[:, 1].flatten().A[0], y_vector.T.flatten().A[0], s = 2, c = 'red')
    plt.show()

if __name__ == '__main__':
    line_regress_local_weight()
