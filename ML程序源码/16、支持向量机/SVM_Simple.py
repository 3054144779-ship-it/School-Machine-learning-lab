from numpy import *
import matplotlib.pyplot as plt

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
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

def SMO_simple(matrix_data_input, labels_class, C, tolerance, iterator_max):
    """
    desc:无
    args:
            matrix_data_input:数据集
            labels_class:类别标签
            C:松弛变量(常量值),允许有些数据点可以处于分隔面的错误一侧;控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重;可以通过调节该参数达到不同的结果
            tolerance:容错率(是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率)
            iterator_max:退出前最大的循环次数
    returns:
            b:模型的常量值
            alphas:拉格朗日乘子
    """
    matrix_data = mat(matrix_data_input)
    matrix_labels = mat(labels_class).transpose()   # 矩阵转置和.T一样的功能
    m, n = shape(matrix_data)
    """
    初始化b和alphas(alpha有点类似权重值)
    """
    b = 0
    alphas = mat(zeros((m, 1)))
    """
    没有任何alpha改变的情况下遍历数据的次数
    """
    iterator = 0
    while (iterator < iterator_max):
        """
        记录alpha是否已经进行优化,每次循环时设为0,然后再对整个集合顺序遍历
        """
        alpha_pairs_changed = 0
        for i in range(m):
            """
            预测的类别y = w^Tx[i] + b;其中因为w = Σ(1~n) a[n]*lable[n]*x[n]
            """
            fx_i = float((multiply(alphas, matrix_labels)).T * (matrix_data * matrix_data[i, :].T)) + b
            E_i = fx_i - float(matrix_labels[i])    # 预测结果与真实结果比对,计算误差E_i
            """
            约束条件(KKT条件是解决最优化问题的时用到的一种方法;这里提到的最优化问题通常是指对于给定的某一函数,求其在指定作用域上的全局最小值)
            0<=alphas[i]<=C,但由于0和C是边界值,我们无法进行优化,因为需要增加一个alphas和降低一个alphas;
            表示发生错误的概率:matrix_labels[i] * E_i,如果超出了tolerance,才需要优化;至于正负号,我们考虑绝对值就对了
            检验训练样本(xi,yi)是否满足KKT条件:
            yi * f(i) >= 1 and alpha = 0(超出边界)
            yi * f(i) == 1 and 0 < alpha < C(在边界上)
            yi * f(i) <= 1 and alpha = C(在边界内)
            """
            if (((matrix_labels[i] * E_i < - tolerance) and (alphas[i] < C)) or ((matrix_labels[i] * E_i > tolerance) and (alphas[i] > 0))):    # 如果满足优化的条件,就随机选取非i的一个点,进行优化比较
                j = select_j_rand(i, m)
                fx_j = float((multiply(alphas, matrix_labels)).T * (matrix_data * matrix_data[j, :].T)) + b # 预测j的结果
                E_j = fx_j - float(matrix_labels[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                """
                L和H用于将alphas[j]调整到0-C之间;如果L==H,就不做任何改变,直接执行continue语句
                matrix_labels[i] != matrix_labels[j]表示异侧,相减;否则是同侧,就相加
                """
                if (matrix_labels[i] != matrix_labels[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                """
                如果相同,就没法优化了
                """
                if L == H:
                    print('L == H')
                    continue
                """
                eta是alphas[j]的最优修改量,如果eta == 0,需要退出for循环的当前迭代过程
                参考<<统计学习方法>>李航-P125~P128<序列最小最优化算法>
                """
                eta = 2.0 * (matrix_data[i, :] * matrix_data[j, :].T) - (matrix_data[i, :] * matrix_data[i, :].T) - (matrix_data[j, :] * matrix_data[j, :].T)
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= matrix_labels[j] * (E_i - E_j) / eta   # 计算出一个新的alphas[j]值
                alphas[j] = clip_alpha(alphas[j], H, L) # 使用辅助函数,以及L和H对其进行调整
                """
                检查alphas[j]是否只是轻微的改变,如果是的话,就退出for循环的当前迭代过程
                """
                if (abs(alphas[j] - alpha_j_old) < 0.00001):
                    print('j没多大变化')
                    continue
                """
                然后alphas[i]和alphas[j]同样进行改变,虽然改变的大小一样,但是改变的方向正好相反
                """
                alphas[i] += matrix_labels[j] * matrix_labels[i] * (alpha_j_old - alphas[j])
                """
                在对alphas[i], alphas[j]进行优化之后,给这两个alphas值设置一个常数b
                w = Σ[1~n] ai * yi * xi => b = yj - Σ[1~n] ai * yi * (xi * xj)
                所以:b1 - b = (y1 - y) - Σ[1~n] yi * (a1 - a) * (xi * x1)
                减两遍的原因:因为是减去Σ[1~n],正好2个变量i和j,所以减2遍
                """
                b1 = b - E_i - matrix_labels[i] * (alphas[i] - alpha_i_old) * (matrix_data[i, :] * matrix_data[i, :].T) - matrix_labels[j] * (alphas[j] - alpha_j_old) * (matrix_data[i, :] * matrix_data[j, :].T)
                b2 = b - E_j - matrix_labels[i] * (alphas[i] - alpha_i_old) * (matrix_data[i, :] * matrix_data[j, :].T) - matrix_labels[j] * (alphas[j] - alpha_j_old) * (matrix_data[j, :] * matrix_data[j, :].T)
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print('迭代:%d,i:%d,被改变的对:%d'%(iterator, i, alpha_pairs_changed))
        """
        在for循环外,检查alpha值是否做了更新,如果在更新则将iterator设为0后继续运行程序;直到更新完毕后,iterator次循环无变化,才推出循环
        """
        if (alpha_pairs_changed == 0):
            iterator += 1
        else:
            iterator += 0
        print('迭代数字:%d' % iterator)
    return b, alphas

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
    vb = array(b)[0]    # b原来是矩阵,先转为数组类型后其数组大小为(1,1),所以后面加[0],变为1
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
    array_data, array_labels = load_data_set('./6.SVM/testSet.txt') # 获取特征和目标变量
    b, alphas = SMO_simple(array_data, array_labels, 0.6, 0.001, 40)    # b是常量值,alphas是拉格朗日乘子
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
