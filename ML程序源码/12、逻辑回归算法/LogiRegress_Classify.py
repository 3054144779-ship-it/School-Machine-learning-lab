from numpy import *
import matplotlib.pyplot as plt

def load_data_set(file_name):
    """
    desc:加载并提取数据
    args:file_name:文件名称,要解析的文件所在磁盘位置
    returns:data_matrix:原始数据的特征,label_vector:原始数据的标签,也就是每条样本对应的类别
    """
    data_matrix = []
    label_vector = []
    fr = open(file_name)
    str_all = fr.readlines()
    fr.close()
    for line in str_all:
        line_array = line.strip().split()   # 1.删除字符串line两端的空白字符 2.以空白字符切割删除完两端空白字符之后的字符串line
        if len(line_array) != 3:
            continue    # 如果line_array的长度不为3,则跳过本次循环
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])   # 为方便计算,这里将X0的值设为1.0,也就是在每行开头添加一个1.0作为X0
        label_vector.append(int(line_array[2]))
    return data_matrix, label_vector

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

def plot_best_fit(data_array, label_vector, weights):
    """
    desc:将得到的数据可视化展示出来
    args:data_array:样本数据的特征,即目标变量;label_vector:样本数据的类别标签,即目标变量;weights:回归系数向量
    returns:无
    """
    m = shape(data_array)[0]
    cord_1_h = []
    cord_1_v = []
    cord_0_h = []
    cord_0_v = []
    for i in range(m):
        if int(label_vector[i] == 1):
            cord_1_h.append(data_array[i, 1])
            cord_1_v.append(data_array[i, 2])
        else:
            cord_0_h.append(data_array[i, 1])
            cord_0_v.append(data_array[i, 2])
    fig = plt.figure()
    sp = fig.add_subplot(111)
    sp.scatter(cord_1_h, cord_1_v, s=30, c='red', marker='s')
    sp.scatter(cord_0_h, cord_0_v, s=30, c= 'green')
    x1 = arange(-3.0, 3.0, 0.1)
    """
    w0*x0+w1*x1+w2*x2=f(x)=0(f(x)被磨合误差给算到w0,w1,w2上去了?)
    x0:最开始设置为1
    x2=(-w0-w1*x1)/w2
    """
    x2 = (- weights[0] - weights[1] * x1) / weights[2]
    sp.plot(x1, x2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
def simple_test():
    # 1.收集并准备数据
    data_matrix, label_vector = load_data_set('./5.Logistic/TestSet.txt')
    # 2.求得权重矩阵
    data_array = array(data_matrix)
    # weights = grad_ascent(data_array, label_vector)
    # weights = rand_grad_ascent_0(data_array, label_vector)
    weights = rand_grad_ascent_1(data_array, label_vector)
    plot_best_fit(data_array, label_vector, weights)

if __name__ == '__main__':
    simple_test()
