from numpy import *
import matplotlib.pyplot as plt

def sigmoid(x_in):
    return 1.0 / (1 + exp(-x_in))

def grad_ascent(data_matrix_in, labels_class_vector):
    """
    desc:正常的梯度上升法
    args:data_matrix_in:输入数据的特征列表(要转化为2维Numpy数组,每列分别代表每个不同的特征,每行则代表每个训练样本);labels_class_vector:输入数据的类别标签(1*100的行向量,为便于矩阵计算,需要将该行向量转换为列向量)
    returns:array(weights):得到的最佳回归系数
    """
    data_matrix = mat(data_matrix_in)   # 转换为Numpy矩阵
    labels_vector = mat(labels_class_vector).transpose()    # 首先将数组转换为Numpy矩阵,之后将行向量转化为列向量=>矩阵的转置
    m, n = shape(data_matrix) # m:数据量,样本数;n:特征数
    alpha = 0.001   # alpha:向目标移动的步长
    cycles_max = 500    # 迭代次数
    weights = ones((n, 1))   # weights:回归系数,生成一个列数和特征数相同的矩阵,其中的数全都是1
    for k in range(cycles_max):
        h = sigmoid(data_matrix * weights)  # (m*n)*(n*1)=(m*1);乘上单位矩阵的意义:通过公式得到的理论值;这里使用的是矩阵乘法
        error = labels_vector - h   # labels_vector:实际值;此处使用向量相减
        weights = weights + alpha * data_matrix.transpose() * error # (n*m)*(m*1)=(n*1);alpha * data_matrix.transpose() * error:在每列上的一个误差情况,最后得出x1,x2,...,xn的系数的偏移量
    return array(weights)

def rand_grad_ascent_0(data_matrix_in, labels_class_vector):
    """
    desc:随机梯度下降,只使用一个样本点来更新回归系数
    args:data_matrix_in:输入数据的数据特征;labels_class_vector:输入数据的类别标签
    returns:weights:得到的最佳回归系数
    notes:梯度下降优化算法在每次更新数据集时都需要遍历整个数据集,计算复杂度较高;随机梯度下降一次只用一个样本点更新回归系数
    """
    m, n =shape(data_matrix_in)
    alpha = 0.01
    weights = ones(n)   # 初始化长度为n的数组(1*n),元素全为1,长这个样子:[1,1,1,1,...,1]
    for i in range(m):
        h = sigmoid(sum(data_matrix_in[i] * weights))   # sum(data_matrix_in[i] * weights):为了求f(x)的值构成的1*1矩阵,f(x)=a1*x1+b2*x2+...+qn*xn;此处的h是一个具体的数值,不是一个矩阵
        error = labels_class_vector[i] - h  # 计算真实类别与预测类别之间的差值,然后按照该差值调整回归系数
        weights = weights + alpha * error * data_matrix_in[i]   # 
    return  weights

def rand_grad_ascent_1(data_matrix_in, labels_class_in, num_cycles = 150):
    """
    desc:改进的随机梯度下降,使用随机的一个样本来更新回归系数
    args:data_matrix_in:输入数据的数据特征;labels_class_in:输入数据的类别标签;num_cycles:迭代次数
    returns:weights:得到的最佳回归系数
    """
    m, n = shape(data_matrix_in)
    weights = ones(n)   # 创建与输入数据的数据特征列数相同的系数矩阵,所有元素均为1
    # 随机梯度,循环150次,观察是否收敛
    for i in range(num_cycles):
        data_index_list = list(range(m))  # [0, 1, 2, ..., m-1]
        for j in range(m):
            alpha = 4 / (1.0 + i + j) + 0.0001  # i和j的不断增大,导致alpha的值不断减小,但不为0;alpha会随着迭代不断减小,但永远非0,因为后面还有个常数项0.0001
            # 随机产生一个0~len之间的一个值
            rand_index = int(random.uniform(0, len(data_index_list)))   # random.uniform(x, y):将随机生成一个实数,它在范围[x,y]内,x:该范围的最小值,y:该范围的最大值
            h = sigmoid(sum(data_matrix_in[data_index_list[rand_index]] * weights)) # sum(data_matrix_in[i] * weights)为了求f(x)=a1*x1+b2*x2+...+qn*xn的值
            error = labels_class_in[data_index_list[rand_index]] - h
            weights = weights + alpha * error * data_matrix_in[data_index_list[rand_index]]
            del data_index_list[rand_index] # 避免在下次循环中再在data_index_list中抽到相同的元素
    return  weights

def classify_vector(x_in, weights):
    """
    desc:最终的分类函数,根据回归系数和特征向量来计算sigmoid的值,>0.5函数返回1,否则返回0
    args:x_in:特征向量;weights:根据梯度下降/随机梯度下降计算得到的回归系数
    returns:如果result>0.5返回1,否则返回0
    """
    return 1 if ((sigmoid(sum(x_in * weights))) > 0.5) else 0

def load_data_from_file(file_name):
    fr = open(file_name)
    str_all = fr.readlines()
    fr.close()
    data_set = []
    result_set = []
    for str_line in str_all:
        cur_line = str_line.strip().split('\t')
        number_str = len(cur_line)
        # print('number_str:', number_str)
        line_array = []
        for i in range(number_str - 1):
            line_array.append(float(cur_line[i]))
        data_set.append(line_array)
        result_set.append(float(cur_line[-1]))
    return data_set, result_set

def colic_test():
    """
    desc:打开训练集和测试集,
    """
    # 1.获得训练用的数据集
    data_set_train, result_set_train = load_data_from_file('./5.Logistic/horseColicTraining.txt')
    # 2.使用改进后的随机梯度下降算法,求得在此数据集上的最佳回归系数矩阵weights_train
    weights_train = rand_grad_ascent_1(array(data_set_train), result_set_train, 500)
    # 3.获得测试用的数据集
    data_set_test, result_set_test = load_data_from_file('./5.Logistic/horseColicTest.txt')
    # 4.测试
    error_cnts = 0
    number_test = len(result_set_test)
    for i in range(number_test):
        if int(classify_vector(array(data_set_test[i]), weights_train)) != int(result_set_test[i]):
            error_cnts += 1
    error_rate = error_cnts / float(number_test)
    print('本次实验的错误率:%f' % error_rate)
    return error_rate

def horse_test():
    number_test =10
    error_rate_sum = 0
    for k in range(number_test):
        error_rate_sum += colic_test()
    print('经过%d次迭代后,平均错误率:%f' % (number_test, error_rate_sum / float(number_test)))

if __name__ == '__main__':
    horse_test()
