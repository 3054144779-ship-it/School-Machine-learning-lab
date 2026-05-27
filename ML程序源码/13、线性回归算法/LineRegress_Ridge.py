from matplotlib.pyplot import figure
from numpy import *
import matplotlib.pylab as plt

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

def ridge_regress(x_matrix, y_vector, lamda = 0.2):
    """
    desc:该函数实现了给定lamda下的岭回归求解;如果数据的特征比样本点还多,就不能再使用线性回归和局部线性回归了,因为计算(xTx)^(-1)会出现错误;如果特征比样本点还多(n>m),也就是说,输入数据的矩阵x不是满秩矩阵;菲满秩矩阵在求逆时会出现问题;为解决这一问题,使用岭回归,这是第一种缩减方法
    args:x_matrix:样本的特征数据;y_vector:每个样本对应的类别标签,即目标变量,实际值;lamda:引入的一个lamda值,使得矩阵非奇异
    returns:经过岭回归公式计算得到的回归系数
    """
    xTx = x_matrix.T * x_matrix
    denom = xTx + lamda * eye(shape(x_matrix)[1])
    if linalg.det(denom) == 0.0:
        print("该矩阵不可逆!")
        return []
    ws = denom.I * (x_matrix.T * y_vector)
    return ws

def ridge_test(x_matrix, y_vector):
    """
    desc:该函数用于在一组lamda上测试结果
    args:x_matrix:样本数据的特征;y_vector:样本数据的类别标签,即真实数据
    returns:w_matrix:将所有的回归系数输出到一个矩阵并返回
    """
    y_matrix = y_vector.T
    """
    mean(matrix, axis = 0),其中matrix为一个矩阵,axis为参数
    以m*n矩阵举例:
    axis不设置值,对m * n个数求均值,返回一个实数
    axis = 0:压缩行,对各列求均值,返回1 * n矩阵
    axis = 1:压缩列,对各行求均值,返回m * 1矩阵
    """
    y_mean = mean(y_matrix, 0)  # 计算y_matrix的均值
    y_matrix = y_matrix - y_mean    # y_matrix的所有特征减去均值
    x_means = mean(x_matrix, 0) # 标准化x,计算x_matrix平均值
    """
    var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue),其中a为一个矩阵,axis为参数
    以m * n矩阵为例:
    axis不设置值,对m*n个数求均值之后求矩阵中所有元素的方差,返回一个实数
    axis = 0:压缩行,对各列求均值,之后求各列中所有元素的方差,返回1 * n矩阵
    axis = 1:压缩列,对各行求均值,之后求各行中所有元素的方差,返回m * 1矩阵
    """
    x_var = var(x_matrix, 0)    # 然后计算x_matrix的方差
    x_handle = (x_matrix - x_means) / x_var # 所有特征都减去各自的均值并除以方差
    number_test = 30    # 可以在30个不同的lamda下调用函数ridge_regress
    w_matrix = zeros((number_test, shape(x_handle)[1])) # 创建number_test * x_handle的列数的全部数据为0的矩阵
    for i in range(number_test):
        ws = ridge_regress(x_handle, y_matrix, exp(i - 10)) # exp()返回e^x
        w_matrix[i, :] = ws.T
    return w_matrix

def linear_regress_ridge():
    x_matrix, y_vector = load_data_set('./8.Regression/abalone.txt')
    weights_ridge = ridge_test(x_matrix, y_vector)
    figure = plt.figure()
    sp = figure.add_subplot(111)
    sp.plot(weights_ridge)
    plt.show()

if __name__ == '__main__':
    linear_regress_ridge()