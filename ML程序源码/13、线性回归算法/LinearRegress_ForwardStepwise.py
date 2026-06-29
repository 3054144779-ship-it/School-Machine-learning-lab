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

def regularize(x_matrix):
    """
    按列进行规范化
    """
    x_regularize_matrix = x_matrix.copy()
    x_regularize_means = mean(x_regularize_matrix, 0)   # 计算x_regularize_matrix的行平均值向量
    x_regularize_var = var(x_regularize_matrix, 0)  # 计算x_regularize_matrix的行方差向量
    x_regularize_matrix = (x_regularize_matrix - x_regularize_means) / x_regularize_var
    return x_regularize_matrix

def rss_error(y_vector, y_hat):
    """
    desc:返回真实值与预测值误差大小
    args:y_vector:样本的真实值;y_hat:样本的预测值
    returns:一个数字,代表误差
    """
    return ((y_vector.getA() - y_hat.getA())**2).sum()

def standard_linear_regress(x_matrix, y_vector):
    """
    desc:线性回归
    args:x_matrix:输入的样本数据,包含每个样本数据的特征;y_vector:对应于输入数据的类别标签,也就是每个样本对应的目标变量
    returns:ws:回归系数矩阵
    """
    xTx = x_matrix.T * x_matrix # 矩阵乘法的条件:左矩阵的列数=右矩阵的行数
    """
    因为要用到xTx的逆矩阵,所以事先需要确定计算得到的xTx是否可逆,条件是矩阵的行列式不为0
    """
    if linalg.det(xTx) == 0.0:  # 函数linalg.det()用来求得矩阵的行列式,如果矩阵的行列式为0,则该矩阵不可逆,就无法进行接下来的运算
        print("矩阵xTx不可逆!")
        return []
    ws = xTx.I * (x_matrix.T * y_vector.T)    #求得w的最优解
    return ws

def forward_stepwise_regression(x_matrix, y_vector, eps = 0.001, num_iteration = 100):
    y_matrix = y_vector.T
    """
    也可以规范化y_matrix,但会得到更小的coef
    """
    y_mean = mean(y_matrix, 0)
    y_matrix = y_matrix - y_mean
    x_matrix = regularize(x_matrix)
    m, n = shape(x_matrix)
    return_matrix = zeros((num_iteration, n))
    ws = zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    for i in range(num_iteration):
        # print(ws.T)
        lowest_error = inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_matrix * ws_test
                error_rss = rss_error(y_matrix, y_test)
                if error_rss < lowest_error:
                    lowest_error = error_rss
                    ws_max = ws_test
        ws = ws_max.copy()
        return_matrix[i, :] = ws.T
    return return_matrix
    
def compare():
    x_matrix, y_vector = load_data_set('./8.Regression/abalone.txt')
    ws_fsr = forward_stepwise_regression(x_matrix, y_vector, 0.01, 200)
    print(ws_fsr)
    x_regularize_matrix = regularize(x_matrix)
    y_vector = y_vector - mean(y_vector, 1)
    ws_sr = standard_linear_regress(x_regularize_matrix, y_vector)
    print(ws_sr.T)

if __name__ == '__main__':
    compare()
