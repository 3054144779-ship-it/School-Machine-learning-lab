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
        for i in range(len(cur_line) -1):   # 除去字符串列表cur_line中的最后一列外
            line_array.append(float(cur_line[i]))   # 将数据添加到列表line_array中,每行的训练数据构成一个行向量
        data_array.append(line_array)  # 将训练数据的输入数据部分存储到列表data_matrix中
        label_array.append(float(cur_line[-1]))    # 将每行的最后一个数据,即类别,或叫目标变量存储到列表label_vector中
    data_matrix = mat(data_array)
    label_vector = mat(label_array)
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

def line_regress_best_fit():
    x_matrix, y_vector = load_data_set("./8.Regression/data.txt")
    ws = standard_linear_regress(x_matrix, y_vector)
    if ws == []:
        print("该问题无线性回归解!")
        return
    fig = plt.figure()
    sp = fig.add_subplot(111) # add_subplot(349)函数参数的意思:将画布分成3行4列,图像画在从左到右从上到下第9块
    sp.scatter(x_matrix[:, 1].flatten().A[0], y_vector.T[:, 0].flatten().A[0])   # 使用mat可将序列转换为二维数组;使用函数flatten可将二维数组转换为折叠的一维数组,矩阵.A=getA(),使用A[0]得到ndarray数组(?);scatter:x为x_matrix中的第2列,y为y_vector的第1列
    x_copy = x_matrix.copy()
    x_copy.sort(0)  # 将矩阵x_copy上的点按列排序,如果直线上的点次序混乱,绘图将出现问题
    y_hat_ascent = x_copy * ws
    sp.plot(x_copy[:, 1], y_hat_ascent, color = 'red')
    plt.show()
    y_hat = x_matrix * ws
    k = corrcoef(y_hat.T, y_vector) # 求该模型的好坏,使用corrcoef求预测值和真实值的相关度
    print(k)

if __name__ == '__main__':
    line_regress_best_fit()