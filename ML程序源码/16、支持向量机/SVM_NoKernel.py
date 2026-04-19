from numpy import *
import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))

class struct_operate:
    def __init__(self, matrix_data_input, labels_class, C, tolerance):  # 用相关参数初始化该结构体
        self.x = matrix_data_input
        self.matrix_labels = labels_class
        self.C = C
        self.tolerance = tolerance
        self.m = shape(matrix_data_input)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.e_cache = mat(zeros((self.m, 2)))  # 首列为有效标记

def load_data_set(name_file):
    """
    desc:对文件进行逐行解析,从而得到类标签和整个特征矩阵
    args:name_file:文件名
    returns:matrix_data:特征矩阵,matrix_label:类标签
    """
    matrix_data = []
    matrix_labels = []
    fr = open(name_file)
    str_all = fr.readlines()
    fr.close()
    for line in str_all:
        array_line = line.strip().split('\t')
        matrix_data.append([float(array_line[0]), float(array_line[1])])
        matrix_labels.append(float(array_line[2]))
    return matrix_data, matrix_labels

def select_j_rand(i, m):
    """
    desc:随机选择一个整数
    args:i:第一个alpha的下标;m:所有alpha的数目
    returns:j:返回一个不为i的随机数,在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clip_alpha(aj, H, L):
    """
    desc:调整aj的值,使aj处于L<=aj<=H
    args:aj:目标值;H:最大值;L:最小值
    returns:aj:目标值
    """
    aj = min(aj, H)
    aj = max(L, aj)
    return aj

def calculate_E_k(struct_operator, k):
    """
    desc:该过程在完整版的SMO算法中出现次数较多,因此将其单独作为一个方法
    args:struct_operator:struct_operate对象;k:具体某一行
    returns:E_k:预测结果与真实结果比对,即计算误差
    """
    # 采用更标准的 .item() 来提取1x1矩阵中的标量值，完美消除警告
    fx_k_matrix = (multiply(struct_operator.alphas, struct_operator.matrix_labels)).T * (struct_operator.x * struct_operator.x[k].T)
    fx_k = fx_k_matrix.item() + struct_operator.b
    E_k = fx_k - struct_operator.matrix_labels[k].item()
    return E_k

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
    if  (((struct_operator.matrix_labels[i].item() * E_i < - struct_operator.tolerance) and (struct_operator.alphas[i].item() < struct_operator.C)) or ((struct_operator.matrix_labels[i].item() * E_i > struct_operator.tolerance) and (struct_operator.alphas[i].item() > 0))):
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
        # 用 .item() 提取数值标量，防止成为矩阵引发警告
        value_eta = (- eta * eta.T).item()
        if value_eta >= 0:
            print('value_eta >= 0')
            return 0
        struct_operator.alphas[j] -= struct_operator.matrix_labels[j].item() * (E_i - E_j) / value_eta # 计算出一个新的alphas[j]值
        struct_operator.alphas[j] = clip_alpha(struct_operator.alphas[j].item(), H, L)
        update_E_k(struct_operator, j)  # 更新误差缓存
        if abs(struct_operator.alphas[j] - alpha_j_old) < 0.00001:
            print('j没多大变化')
            return 0
        struct_operator.alphas[i] += struct_operator.matrix_labels[j].item() * struct_operator.matrix_labels[i].item() * (alpha_j_old.item() - struct_operator.alphas[j].item())    # alphas[i]和alphas[j]同样进行改变,虽然改变的大小一样,但是改变的方向正好相反
        update_E_k(struct_operator, i)  # 更新误差缓存
        """
        在对alphas[i], alphas[j]进行优化之后,给这两个alphas值设置一个常数b
        w = Σ[1~n] ai * yi * xi => b = yj - Σ[1~n] ai * yi * (xi * xj)
        所以:b1 - b = (y1 - y) - Σ[1~n] yi * (a1 - a) * (xi * x1)
        减两遍的原因:因为是减去Σ[1~n],正好2个变量i和j,所以减2遍
        """
        # 用 .item() 提取数值标量
        b1 = (struct_operator.b - E_i - struct_operator.matrix_labels[i].item() * (struct_operator.alphas[i].item() - alpha_i_old.item()) * (struct_operator.x[i] * struct_operator.x[i].T) - struct_operator.matrix_labels[j].item() * (struct_operator.alphas[j].item() - alpha_j_old.item()) * (struct_operator.x[i] * struct_operator.x[j].T)).item()
        b2 = (struct_operator.b - E_j - struct_operator.matrix_labels[i].item() * (struct_operator.alphas[i].item() - alpha_i_old.item()) * (struct_operator.x[i] * struct_operator.x[j].T) - struct_operator.matrix_labels[j].item() * (struct_operator.alphas[j].item() - alpha_j_old.item()) * (struct_operator.x[j] * struct_operator.x[j].T)).item()
        if (0 < struct_operator.alphas[i].item()) and (struct_operator.C > struct_operator.alphas[i].item()):
            struct_operator.b = b1
        elif (0 < struct_operator.alphas[j].item()) and (struct_operator.C > struct_operator.alphas[j].item()):
            struct_operator.b = b2
        else:
            struct_operator.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def SMO_perfect(matrix_data_input, labels_class, C, tolerance, iterator_max):
    """
    desc:完整SMO算法外循环,与SMO_simple有些类似,但这里的循环退出条件更多一些
    args:
            matrix_data_input:数据集
            labels_class:类别标签
            C:松弛变量(常量值),允许有些数据点可以处于分隔面的错误一侧;控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重;可以通过调节该参数达到不同的结果
            tolerance:容错率
            iterator_max:退出前最大的循环次数
    returns:b:模型的常量值;alphas:拉格朗日乘子
    """
    struct_operator = struct_operate(mat(matrix_data_input), mat(labels_class).T, C, tolerance) # 创建一个struct_operate对象
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

def plot_figure_SVM(matrix_x, matrix_y, ws, b, alphas):
    """
    http://blog.csdn.net/maoersong/article/details/24315633
    http://www.cnblogs.com/JustForCS/p/5283489.html
    http://blog.csdn.net/kkxgx/article/details/6951959
    """
    mx = mat(matrix_x)
    my = mat(matrix_y)
    vb = b    # 【已修改】因为现在b已经是普通的标量数字了，不需要再 array(b)[0]
    figure = plt.figure()
    sp = figure.add_subplot(111)
    sp.scatter(mx[:, 0].flatten().A[0], mx[:, 1].flatten().A[0])    # 注意flatten的用法
    x = arange(-1.0, 10.0, 0.1)    # x最大值,最小值根据原数据集mx[:, 0]的大小而定
    y = (-vb - ws[0, 0] * x) / ws[1, 0]   # 根据x.w + b = 0 得到,其式子展开为w0.x1 + w1.x2 + b = 0,x2就是y值
    sp.plot(x, y)
    for i in range(shape(my[0, :])[1]):
        if my[0, 1] > 0:
            sp.plot(mx[i, 0], mx[i, 1], 'cx')
        else:
            sp.plot(mx[i, 0], mx[i, 1], 'kp')
    """
    找到支持向量,并在图中标红
    """
    for i in range(100):
        if alphas[i] > 0.0:
            sp.plot(mx[i, 0], mx[i, 1], 'ro')
    plt.show()

if __name__ == '__main__':
    # 动态拼接数据文件的绝对路径
    data_path = os.path.join(current_dir, '6.SVM', 'testSet.txt')
    array_data, array_labels = load_data_set(data_path) # 获取特征和目标变量
    b, alphas = SMO_perfect(array_data, array_labels, 0.6, 0.001, 40)    # b是常量值,alphas是拉格朗日乘子
    print('b = ',b)
    print('alphas[alphas > 0] = ', alphas[ alphas > 0])
    print('shape(alphas[alphas > 0]) = ',shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0:
            print(array_data[i], array_labels[i])
    """
    画图
    """
    ws = calculate_ws(alphas, array_data, array_labels)
    plot_figure_SVM(array_data, array_labels, ws, b, alphas)