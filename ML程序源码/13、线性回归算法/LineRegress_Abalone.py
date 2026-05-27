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
    if linalg.det(xTx) == 0.0:
        print("该矩阵为奇异矩阵,不可逆")
        return []
    ws = xTx.I * (x_matrix.T * (weights * y_matrix))    # 计算出回归系数的一个估计
    return test_point * ws

def lwlr_test(test_matrix, x_matrix, y_vector, k = 1.0):
    """
    desc:测试局部加权线性回归,对数据集中每个点调用函数lwlr()
    args:test_matrix:测试所用的所有样本点;x_matrix:样本的特征数据;y_vector:每个样本对应的类别标签,即目标变量;k:控制核函数的衰减速率
    returns:y_hat:预测点的估计值
    """
    m = shape(test_matrix)[0]   # 得到样本点的总数
    y_hat = mat(zeros(m))    # 构建一个全都是0的1*m矩阵
    """
    循环所有的数据点,并将lwlr运用于所有的数据点
    """
    for i in range(m):
        # print(shape(test_matrix[i]))
        # print(shape(x_matrix))
        # print(shape(y_vector))
        y_hat[0, i] = lwlr(test_matrix[i], x_matrix, y_vector, k)
    return y_hat    #返回估计值

def rss_error(y_vector, y_hat):
    """
    desc:返回真实值与预测值误差大小
    args:y_vector:样本的真实值;y_hat:样本的预测值
    returns:一个数字,代表误差
    """
    return ((y_vector.getA() - y_hat.getA())**2).sum()
    # return ((y_vector - y_hat) * (y_vector - y_hat).T)

def abalone_test():
    """
    desc:预测鲍鱼的年龄
    args:None
    returns:None
    """
    x_abalone, y_abalone = load_data_set('./8.Regression/abalone.txt')  # 加载数据
    # print(x_abalone)
    # print(shape(y_abalone[0:99]))
    """
    使用不同的核进行预测
    """
    y_hat_01_0_99 = lwlr_test(x_abalone[0:99], x_abalone[0:99], y_abalone[0, 0:99], 0.1)
    y_hat_1_0_99 = lwlr_test(x_abalone[0:99], x_abalone[0:99], y_abalone[0, 0:99], 1)
    y_hat_10_0_99 = lwlr_test(x_abalone[0:99], x_abalone[0:99], y_abalone[0, 0:99], 10)
    """
    打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    """
    print(shape(y_abalone[0, 0:99]))
    print(shape(y_hat_01_0_99))
    print("y_hat_01_0_99的误差:", rss_error(y_abalone[0, 0:99], y_hat_01_0_99))
    print("y_hat_1_0_99的误差:", rss_error(y_abalone[0, 0:99], y_hat_1_0_99))
    print("y_hat_10_0_99的误差:", rss_error(y_abalone[0, 0:99], y_hat_10_0_99))
    """
    使用不同的核预测值进行预测
    """
    y_hat_01_100_199 = lwlr_test(x_abalone[100:199], x_abalone[0:99], y_abalone[0, 0:99], 0.1)
    y_hat_1_100_199 = lwlr_test(x_abalone[100:199], x_abalone[0:99], y_abalone[0, 0:99], 1)
    y_hat_10_100_199 = lwlr_test(x_abalone[100:199], x_abalone[0:99], y_abalone[0, 0:99], 10)
    """
    打印出不同的核预测值与测试数据集上的真实值之间的误差大小
    """
    print("y_hat_01_100_199的误差:", rss_error(y_abalone[0, 100:199], y_hat_01_100_199))
    print("y_hat_1_100_199的误差:", rss_error(y_abalone[0, 100:199], y_hat_1_100_199))
    print("y_hat_10_100_199的误差:", rss_error(y_abalone[0, 100:199], y_hat_10_100_199))
    """
    使用简单的线性回归进行预测,与上面的计算进行比较
    """
    ws_standard = standard_linear_regress(x_abalone[0:99], y_abalone[0, 0:99])
    y_hat_standard = x_abalone[100:199] * ws_standard
    print("y_hat_standard的误差:", rss_error(y_abalone[0, 100:199], y_hat_standard.T))

if __name__ == '__main__':
    abalone_test()
