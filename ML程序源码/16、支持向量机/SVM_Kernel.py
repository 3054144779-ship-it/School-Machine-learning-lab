from numpy import *
import matplotlib.pyplot as plt

def kernel_transform(matrix_data_input, matrix_data_input_no_i, k_tupple):  # 计算内核或将数据转换到更高维度的空间
    """
    desc:核转换函数
    args:matrix_data_input:输入数据集;matrix_data_input_no_i:输入数据集的第i行的数据;k_tupple:核函数信息
    returns:
    """
    m, n =shape(matrix_data_input)
    matrix_k = mat(zeros((m, 1)))
    if k_tupple[0] == 'lin':    # 线性内核:(m, n)*(n, 1)=(m, 1)
        matrix_k = matrix_data_input * matrix_data_input_no_i.T
    elif k_tupple[0] == 'rbf':
        for j in range(m):
            delta_row = matrix_data_input[j, :] - matrix_data_input_no_i
            matrix_k[j] = delta_row * delta_row.T
        matrix_k = exp(matrix_k / (-1 * k_tupple[1] ** 2))  # 径向基函数的高斯版本;在Numpy中的除法是元素方面的,而不像matlab那样是矩阵的
    else:
        raise NameError('出现问题F:此内核无法被识别')
    return  matrix_k

class struct_operate:
    def __init__(self, matrix_data_input, labels_class, C, tolerance, k_tupple):  # 用相关参数初始化该结构体
        """
        args:
                matrix_data_input:数据集
                labels_class:类别标签
                C:松弛变量(常量值),允许有些数据点可以处于分隔面的错误一侧;控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重;可以通过调节该参数达到不同的结果
                tolerance:容错率
                k_tupple:包含核函数信息的元组
        """
        self.x = matrix_data_input
        self.matrix_labels = labels_class
        self.C = C
        self.tolerance = tolerance
        self.m = shape(matrix_data_input)[0]    # 数据的行数
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.e_cache = mat(zeros((self.m, 2)))  # 误差缓存,第一列给出的是e_cache是否有效的标志位,第二列给出的是实际的E值
        self.matrix_k = mat(zeros((self.m, self.m)))    # m行m列的矩阵
        for i in range(self.m):
            self.matrix_k[:, i] = kernel_transform(self.x, self.x[i], k_tupple)

def load_data_set(name_file):
    """
    desc:对文件进行逐行解析,从而得到类标签和整个数据矩阵
    args:name_file:文件名
    returns:matrix_data:数据矩阵;matrix_labels:类标签
    """
    matrix_data = []
    matrix_labels = []
    fr = open(name_file)
    str_all = fr.readlines()
    fr.close()
    for str_line in str_all:
        array_line = str_line.strip().split('\t')
        matrix_data.append([float(array_line[0]), float(array_line[1])])
        matrix_labels.append(float(array_line[2]))
    return matrix_data, matrix_labels

def calculate_E_k(struct_operator, k):
    """
    desc:该过程在完整版的SMO算法中出现次数较多,因此将其单独作为一个方法
    args:struct_operator:struct_operate对象;k:具体某一行
    returns:E_k:预测结果与真实结果比对,即计算误差
    """
    fx_k = (multiply(struct_operator.alphas, struct_operator.matrix_labels)).T * (struct_operator.x * struct_operator.x[k].T) + struct_operator.b
    E_k = fx_k - float(struct_operator.matrix_labels[k])
    return E_k

def select_j_rand(i, m):
    """
    desc:随机选择一个整数
    args:i:第一个alpha的下标;m:所有alpha的数目
    returns:j:返回一个不为i的随机数,在0~m之间的整数值
    """
    j = i
    while j == i:
        j = random.randint(0, m - 1)
    return j

def select_j(i, struct_operator, E_i):  # 这是第二选择-启发式方法并计算E_j
    """
    desc:返回最优的j和E_j;内循环的启发式方法;选择第二个(内循环)alpha的alpha值;这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大步长;该函数的误差与第一个alpha值Ei和下标i有关
    args:i:具体的第i行;struct_operator:struct_operate对象;E_i:预测结果与真实结果比对,计算误差E_i
    returns:j:随机选出的第j行;E_j:预测结果与真实结果比对,计算误差E_j
    """
    k_max = -1
    delta_E_max = 0
    E_j = 0
    """
    首先将输入值E_i在缓存中设置成为有效的;这里的有效意味着它已经计算好了
    """
    struct_operator.e_cache[i] = [1, E_i]
    list_valid_e_cache = nonzero(struct_operator.e_cache[:, 0].A)[0]    # nonzero:返回非0的:行列值;取行的list;非零E值的行的list列表,所对应的alpha值
    if (len(list_valid_e_cache)) > 1:
        for k in list_valid_e_cache:    # 在所有的值上进行循环,并选择其中使得改变最大的那个值
            if k == i:  # 对于i不要计算
                continue
            E_k = calculate_E_k(struct_operator, k) # 求E_k误差:预测值 - 真实值的差
            delta_E = abs(E_i - E_k)
            if delta_E > delta_E_max:
                k_max = k
                delta_E_max = delta_E
                E_j = E_k
        return k_max, E_j
    else:   # 如果是第一次循环,则随机选择一个alpha值
        j = select_j_rand(i, struct_operator.m)
        E_j = calculate_E_k(struct_operator, j) # 求E_k误差:预测值 - 真实值的差
        return j, E_j

def update_E_k(struct_operator, k):
    """
    desc:在所有alpha都已改变之后在缓存内更新新的值
    args:struct_operator:struct_operate对象;k:某一列的行号
    returns:无
    """
    E_k = calculate_E_k(struct_operator, k) # 求误差:预测值-真实值的差
    struct_operator.e_cache[k] = [1, E_k]

def clip_alpha(aj, H, L):
    """
    desc:调整aj的值,使aj处于L<=aj<=H
    args:aj:目标值;H:最大值;L:最小值
    returns:aj:目标值
    """
    aj = min(aj, H)
    aj = max(L, aj)
    return aj

def inner_loop(i, struct_operator):
    """
    desc:内循环代码
    args:i:具体的某一行;struct_operator:struct_operate对象
    returns:0:找不到最优的值;1:找到了最优的值,并且struct_operator.e_cache到缓存中
    """
    E_i = calculate_E_k(struct_operator, i) # 求E_i误差:预测值 - 真实值的差
    """
    约束条件(KKT条件是解决最优化问题的时用到的一种方法;这里提到的最优化问题通常是指对于给定的某一函数,求其在指定作用域上的全局最小值)
    0<=alphas[i]<=C,但由于0和C是边界值,我们无法进行优化,因为需要增加一个alphas和降低一个alphas;
    表示发生错误的概率:matrix_labels[i] * E_i,如果超出了tolerance,才需要优化;至于正负号,我们考虑绝对值就对了
    检验训练样本(xi,yi)是否满足KKT条件:
    yi * f(i) >= 1 and alpha = 0(超出边界)
    yi * f(i) == 1 and 0 < alpha < C(在边界上)
    yi * f(i) <= 1 and alpha = C(在边界内)
    """
    if  (((struct_operator.matrix_labels[i] * E_i < - struct_operator.tolerance) and (struct_operator.alphas[i] < struct_operator.C)) or ((struct_operator.matrix_labels[i] * E_i > struct_operator.tolerance) and (struct_operator.alphas[i] > 0))):
        j, E_j = select_j(i, struct_operator, E_i)
        alpha_i_old = struct_operator.alphas[i].copy()
        alpha_j_old = struct_operator.alphas[j].copy()
        """
        L和H用于将alphas[j]调整到0-C之间;如果L==H,就不做任何改变,直接return 0
        """
        if struct_operator.matrix_labels[i] != struct_operator.matrix_labels[j]:
            L = max(0, struct_operator.alphas[j] - struct_operator.alphas[i])
            H = min(struct_operator.C, struct_operator.C + struct_operator.alphas[j] - struct_operator.alphas[i])
        else:
            L = max(0, struct_operator.alphas[j] + struct_operator.alphas[i] - struct_operator.C)
            H = min(struct_operator.C, struct_operator.alphas[j] + struct_operator.alphas[i])
        if L == H:
            print('L == H')
            return 0
        """
        value_eta是alphas[j]的最优修改量,如果value_eta == 0,需要退出for循环的当前迭代过程;参考<<统计学习方法>>李航-P125~P128<序列最小最优化算法>
        """
        eta = struct_operator.x[i] - struct_operator.x[j]
        value_eta = - eta * eta.T
        if value_eta >= 0:
            print('value_eta >= 0')
            return 0
        struct_operator.alphas[j] -= struct_operator.matrix_labels[j] * (E_i - E_j) / value_eta # 计算出一个新的alphas[j]值
        struct_operator.alphas[j] = clip_alpha(struct_operator.alphas[j], H, L)
        update_E_k(struct_operator, j)  # 更新误差缓存
        if abs(struct_operator.alphas[j] - alpha_j_old) < 0.00001:
            print('j没多大变化')
            return 0
        struct_operator.alphas[i] += struct_operator.matrix_labels[j] * struct_operator.matrix_labels[i] * (alpha_j_old - struct_operator.alphas[j])    # alphas[i]和alphas[j]同样进行改变,虽然改变的大小一样,但是改变的方向正好相反
        update_E_k(struct_operator, i)  # 更新误差缓存
        """
        在对alphas[i], alphas[j]进行优化之后,给这两个alphas值设置一个常数b
        w = Σ[1~n] ai * yi * xi => b = yj - Σ[1~n] ai * yi * (xi * xj)
        所以:b1 - b = (y1 - y) - Σ[1~n] yi * (a1 - a) * (xi * x1)
        减两遍的原因:因为是减去Σ[1~n],正好2个变量i和j,所以减2遍
        """
        b1 = struct_operator.b - E_i - struct_operator.matrix_labels[i] * (struct_operator.alphas[i] - alpha_i_old) * (struct_operator.x[i] * struct_operator.x[i].T) - struct_operator.matrix_labels[j] * (struct_operator.alphas[j] - alpha_j_old) * (struct_operator.x[i] * struct_operator.x[j].T)
        b2 = struct_operator.b - E_j - struct_operator.matrix_labels[i] * (struct_operator.alphas[i] - alpha_i_old) * (struct_operator.x[i] * struct_operator.x[j].T) - struct_operator.matrix_labels[j] * (struct_operator.alphas[j] - alpha_j_old) * (struct_operator.x[j] * struct_operator.x[j].T)
        if (0 < struct_operator.alphas[i]) and (struct_operator.C > struct_operator.alphas[i]):
            struct_operator.b = b1
        elif (0 < struct_operator.alphas[j]) and (struct_operator.C > struct_operator.alphas[j]):
            struct_operator.b = b2
        else:
            struct_operator.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def SMO_perfect(matrix_data_input, labels_class, C, tolerance, iterator_max, k_tupple = ('lin', 0)):
    """
    desc:完整SMO算法外循环,与SMO_simple有些类似,但这里的循环退出条件更多一些
    args:
            matrix_data_input:数据集
            labels_class:类别标签
            C:松弛变量(常量值),允许有些数据点可以处于分隔面的错误一侧;控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重;可以通过调节该参数达到不同的结果
            tolerance:容错率
            iterator_max:退出前最大的循环次数
            k_tupple:包含核函数信息的元组
    returns:b:模型的常量值;alphas:拉格朗日乘子
    """
    struct_operator = struct_operate(mat(matrix_data_input), mat(labels_class).T, C, tolerance, k_tupple) # 创建一个struct_operate对象
    iterator = 0
    b_entire_set = True
    alpha_pairs_changed = 0
    """
    循环遍历: 循环iterator_max次并且(alpha_pairs_changed存在可以改变或所有行遍历一遍)
    循环迭代结束或者循环遍历所有alpha后,alpha_pairs还是没变化
    """
    while (iterator < iterator_max) and ((alpha_pairs_changed > 0) or b_entire_set):
        alpha_pairs_changed = 0
        """
        当b_entire_set = true 或者 非边界alpha_pairs没有了时:就开始寻找alpha_pairs,然后决定是否要进行else
        """
        if b_entire_set:
            """
            在数据集上遍历所有可能的alpha
            """
            for i in range(struct_operator.m):
                alpha_pairs_changed += inner_loop(i, struct_operator)   # 是否存在alpha_pair,存在就+1
                print('满集,迭代器:%d,i:%d,已改变的对:%d' % (iterator, i, alpha_pairs_changed))
            iterator += 1
        else:   # 对已存在alpha_pair,选出非边界的alpha值,进行优化
            non_bound_i = nonzero((struct_operator.alphas.A > 0) * (struct_operator.alphas.A < C))[0]   # 遍历所有的非边界alpha值,也就是不在边界0或C上的值
            for i in non_bound_i:
                alpha_pairs_changed += inner_loop(i, struct_operator)
                print('非边界,迭代器:%d,i:%d,已改变的对:%d' % (iterator, i, alpha_pairs_changed))
            iterator += 1
        if b_entire_set:    # 如果找到alpha_pair,就优化非边界alpha值,否则,就重新进行寻找,如果寻找一遍,遍历所有的行还是没找到,就退出循环
            b_entire_set = False    # 切换整个集合循环
        elif alpha_pairs_changed == 0:
            b_entire_set = True
        print('迭代数:%d' % iterator)
    return struct_operator.b, struct_operator.alphas

def calculate_ws(alphas, array_data, labels_class):
    """
    desc:基于alphas计算w值
    args:alphas:拉格朗日乘子;array_data:特征数据集;labels_class:目标变量数据集
    returns:wc:回归系数
    """
    matrix_x = mat(array_data)
    matrix_labels = mat(labels_class).T
    m, n = shape(matrix_x)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * matrix_labels[i], matrix_x[i, :].T)
    return w

def test_kernel_rbf(k1 = 1.3):
    array_data_train, array_labels_train = load_data_set('./6.SVM/testSetRBF.txt')
    b, alphas = SMO_perfect(array_data_train, array_labels_train, 200, 0.0001, 10000, ('rbf', k1))   # C=200十分重要
    matrix_data_train = mat(array_data_train)
    matrix_labels_train = mat(array_labels_train).T
    index_SV = nonzero(alphas.A > 0)[0]
    matrix_data_SV = matrix_data_train[index_SV]   # 获得仅支持向量的矩阵
    matrix_labels_SV = matrix_labels_train[index_SV]
    print("有%d个支持向量" % shape(matrix_data_SV)[0])
    m_train, n_train = shape(matrix_data_train)
    count_train_error = 0
    for i in range(m_train):
        E_value_kernel_train = kernel_transform(matrix_data_SV, matrix_data_train[i, :], ('rbf', k1))
        predict_train = E_value_kernel_train.T * multiply(matrix_labels_SV, alphas[index_SV]) + b
        if sign(predict_train) != sign(array_labels_train[i]):
            count_train_error += 1
    print('训练错误率:%f' % (float(count_train_error) / m_train))
    array_data_test, array_labels_test = load_data_set('./6.SVM/testSetRBF2.txt')
    count_test_error = 0
    matrix_data_test = mat(array_data_test)
    matrix_labels_test = mat(array_labels_test)
    m_test, n_test = shape(array_data_test)
    for i in range(m_test):
        E_value_kernel_test = kernel_transform(matrix_data_SV, matrix_data_test[i, :], ('rbf', k1))
        predict_test = E_value_kernel_test.T * multiply(matrix_labels_SV, alphas[index_SV]) + b
        if sign(predict_test) != sign(array_labels_test[i]):
            count_test_error += 1
    print('测试错误率:%f' % (float(count_test_error) / m_test))

def image2vector(name_file):
    vector_return = zeros((1, 1024))
    fr = open(name_file)
    str_all = fr.readlines()
    fr.close()
    for i in range(32):
        for j in range(32):
            vector_return[0, 32 * i + j] = int(str_all[i][j])
    return vector_return

def load_images(name_directory):
    from os import listdir
    labels_hw = []
    print(name_directory)
    list_file = listdir(name_directory) # 加载数据集
    m = len(list_file)
    matrix_file = zeros((m, 1024))
    for i in range(m):
        str_name_file = list_file[i]
        str_file = str_name_file.split('.')[0]  # 去掉.txt
        str_number_class = int(str_file.split('_')[0])
        if str_number_class == 9:
            labels_hw.append(-1)
        else:
            labels_hw.append(1)
        matrix_file[i, :] = image2vector("%s/%s" % (name_directory, str_name_file))
    return matrix_file, labels_hw

def test_digits(k_tupple = ('rbf', 10)):
    array_data_train, array_labels_train = load_images('./6.SVM/trainingDigits')  # 导入训练数据
    b, alphas = SMO_perfect(array_data_train, array_labels_train, 200, 0.0001, 10000, k_tupple)   # C=200十分重要
    matrix_data_train = mat(array_data_train)
    matrix_labels_train = mat(array_labels_train).T
    index_SV = nonzero(alphas.A > 0)[0]
    index_SV = nonzero(alphas.A > 0)[0]
    matrix_data_SV = matrix_data_train[index_SV]   # 获得仅支持向量的矩阵
    matrix_labels_SV = matrix_labels_train[index_SV]
    print("有%d个支持向量" % shape(matrix_data_SV)[0])
    m_train, n_train = shape(matrix_data_train)
    count_train_error = 0
    for i in range(m_train):
        E_value_kernel_train = kernel_transform(matrix_data_SV, matrix_data_train[i, :], k_tupple)
        predict_train = E_value_kernel_train.T * multiply(matrix_labels_SV, alphas[index_SV]) + b
        if sign(predict_train) != sign(array_labels_train[i]):
            count_train_error += 1
    print('训练错误率:%f' % (float(count_train_error) / m_train))
    array_data_test, array_labels_test = load_images('./6.SVM/testDigits')
    count_test_error = 0
    matrix_data_test = mat(array_data_test)
    matrix_labels_test = mat(array_labels_test)
    m_test, n_test = shape(array_data_test)
    for i in range(m_test):
        E_value_kernel_test = kernel_transform(matrix_data_SV, matrix_data_test[i, :], k_tupple)
        predict_test = E_value_kernel_test.T * multiply(matrix_labels_SV, alphas[index_SV]) + b
        if sign(predict_test) != sign(array_labels_test[i]):
            count_test_error += 1
    print('测试错误率:%f' % (float(count_test_error) / m_test))

if __name__ == '__main__':
    # test_kernel_rbf()
    test_digits(('rbf', 10))
