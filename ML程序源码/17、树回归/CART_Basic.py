from numpy import *

def load_data_set(name_file):   # 默认解析的数据是用tab分隔,并且是数值类型
    """
    desc:解析每一行,并转化为float类型;该函数读取一个以 tab 键为分隔符的文件;然后将每行的内容保存成一组浮点数
    args:name_file:文件名
    returns:matrix_data:每一行的数据集;数组类型
    notes:假定最后一列是结果值
    """
    matrix_data = []
    fr = open(name_file)
    str_all = fr.readlines()
    fr.close()
    for str_line in str_all:
        str_current = str_line.strip().split('\t')
        list_float = [float(x) for x in str_current]    # 将每行转换成浮点数
        matrix_data.append(list_float)
    return matrix_data

def binary_split_data_set(matrix_data, feature, value):
    """
    desc:将数据集,按照feature列的value进行二元切分;在给定特征和特征值的情况下,该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回
    args:matrix_data:数据集;feature:待切分的特征列;value:特征列要比较的值
    returns:matrix_0:<=value的数据集在左边;matrix_1:>value的数据集在右边
    """
    """
    matrix_data[:, feature]取每一行中,第1列的值(从0开始算)
    nonzero(matrix_data[:, feature] > value):返回结果为True行的index下标
    """
    matrix_0 = matrix_data[nonzero(matrix_data[:, feature] <= value)[0], :]
    matrix_1 = matrix_data[nonzero(matrix_data[:, feature] > value)[0], :]
    return matrix_0, matrix_1

def regress_leaf(matrix_data):
    """
    desc:返回每一个叶子结点的均值
    个人理解:regress_leaf:产生叶节点的函数,就是求均值,即用聚类中心点来代表这类数据
    """
    return mean(matrix_data[:, -1])

def regress_error(matrix_data):
    """
    计算总方差 = 方差 * 样本数
    个人理解:求这组数据的方差,即通过决策树划分,可以让靠近的数据分到同一类中去
    """
    return var(matrix_data[:, -1]) * (shape(matrix_data)[0])  # shape(matrix_data)[0]:行数

def choose_best_split(matrix_data, type_leaf = regress_leaf, type_error = regress_error, ops =(1, 4)):
    """
    desc:1.用最佳方式切分数据集2.生成相应的叶节点
    args:matrix_data:加载的原始数据集;type_leaf:建立叶子点的函数;type_error:误差计算函数(求总方差);ops:(容许误差下降值,切分的最少样本数)
    returns:index_best:特征的索引坐标;value_best:切分的最佳值
    notes:
            ops=(1,4),非常重要,因为它决定了决策树划分停止的threshold值,被称为预剪枝(prepruning),其实也就是用于控制函数的停止时机;之所以这样说,是因为它防止决策树的过拟合,所以当误差的下降值小于tolS,或划分后的集合size小于tolN时,选择停止继续划分
    """
    to1S = ops[0]   # 最小误差下降值,划分后的误差减小小于这个差值,就不用继续划分
    to1N = ops[1]   # 划分最小size,小于,就不继续划分了
    """
    如果数据集的最后一列所有值相等就退出
    """
    if len(set(matrix_data[:, -1].T.tolist()[0])) == 1: # matrix_data[:, -1].T.tolist()[0]:取数据集的最后一列,转置为行向量,然后转换为list,取该list中的第一个元素;如果集合size为1,也就是说全部的数据都是同一个类别,不用继续划分
        return None, type_leaf(matrix_data)
    m, n = shape(matrix_data)   # 计算行列值
    s = type_error(matrix_data) # 无分类误差的总方差和;来自均值的RSS误差的减小驱使最佳特征的选择
    s_best, index_best, value_best = inf, 0, 0  # inf:正无穷大
    """
    循环处理每一列对应的feature值
    """
    for index_feature in range(n - 1):  # 对每个特征
        for value_split in set(matrix_data[:, index_feature].T.tolist()[0]):    # 将某一列全部的数据转换为行,然后设置为list形式
            matrix_0, matrix_1 =binary_split_data_set(matrix_data, index_feature, value_split)  # 对该列进行分组,然后组内的成员的value值进行二元切分
            if (shape(matrix_0)[0] < to1N) or (shape(matrix_1)[0] < to1N):  # 判断二元切分的方式的元素数量是否符合预期
                continue
            s_new = type_error(matrix_0) + type_error(matrix_1) 
            """
            如果二元切分,算出来的误差在可接受范围内,那么就记录切分点,并记录最小误差
            如果划分后误差小于s_best,则说明找到了新的s_best
            """
            if s_new < s_best:
                index_best = index_feature
                value_best = value_split
                s_best = s_new
    """
    判断二元切分的方式的元素误差是否符合预期
    如果(s_best)的减小小于阈值,不做划分
    """
    if (s - s_best) < to1S:
        return None, type_leaf(matrix_data)
    matrix_0, matrix_1 = binary_split_data_set(matrix_data, index_best, value_best)
    """
    对整体的成员进行判断,是否符合预期
    """
    if (shape(matrix_0)[0] < to1N) or (shape(matrix_1)[0] < to1N):  # 如果集合的size小于tolN,当最佳划分后,集合过小,也不划分,产生叶节点
        return None, type_leaf(matrix_data)
    return index_best, value_best

def create_tree(matrix_data, type_leaf = regress_leaf, type_error = regress_error, ops = (1, 4)):   # 假设matrix_data是NumPy Mat类型的,那么我们可以进行array过滤
    """
    desc:获取回归树;递归函数:如果构建的是回归树,该模型是一个常数;果是模型树,其模型是一个线性方程
    args:matrix_data:加载的原始数据集;type_leaf:建立叶子点的函数;type_error:误差计算函数;ops:(容许误差下降值,切分的最少样本数)
    returns:tree_return:决策树最后的结果
    """
    feature, value = choose_best_split(matrix_data, type_leaf, type_error, ops) # 选择最好的切分方式:feature索引值,最优切分值
    if feature is None: # 如果分割达到一个停止条件,那么返回value
        return value
    tree_return = {}
    tree_return['split_index'] = feature
    tree_return['split_value'] = value
    set_l, set_r = binary_split_data_set(matrix_data, feature, value)   # >在右边,<=在左边,分为2个数据集
    """
    递归的进行调用,在左右子树中继续递归生成树
    """
    tree_return['left'] = create_tree(set_l, type_leaf, type_error, ops)
    tree_return['right'] = create_tree(set_r, type_leaf, type_error, ops)
    return tree_return

def is_tree(obj):   # 判断节点是否是一个字典
    """
    desc:测试输入变量是否是一棵树,即是否是一个字典
    args:obj:输入变量
    returns:返回布尔类型的结果;如果obj是一个字典,返回true,否则返回false
    """
    return (type(obj).__name__ == 'dict')

def get_mean(tree): # 计算左右枝丫的均值
    """
    desc:从上往下遍历树直到叶节点为止,如果找到两个叶节点则计算它们的平均值;对tree进行塌陷处理,即返回树平均值
    args:tree:输入的树
    returns:返回树中节点的平均值
    """
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean('left')
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, data_test): # 检查是否适合合并分枝
    """
    desc:从上而下找到叶节点,用测试数据集来判断将这些叶节点合并是否能降低测试误差
    args:tree:待剪枝的树;data_test:剪枝所需要的测试数据
    returns:tree:剪枝完成的树
    """
    """
    判断是否测试数据集没有数据,如果没有,就直接返回tree本身的均值
    """
    if shape(data_test)[0] == 0:
        return get_mean(tree)
    """
    判断分枝是否是dict字典,如果是就将测试数据集进行切分
    """
    if (is_tree(tree['right'])) or (is_tree(tree['left'])):
        set_l, set_r = binary_split_data_set(data_test, tree['split_index'], tree['split_value'])
    """
    如果是左边分枝是字典,就传入左边的数据集和左边的分枝,进行递归
    """
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], set_l)
    """
    如果是右边分枝是字典,就传入左边的数据集和左边的分枝,进行递归
    """
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], set_r)
    
    """
    上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好,对应的值放到对应的节点
    """

    """
    如果左右两边同时都不是dict字典,也就是左右两边都是叶节点,而不是子树了,那么分割测试数据集
    如果正确:那么计算一下总方差和该结果集的本身不分枝的总方差,并进行比较
    如果合并的总方差<不合并的总方差,那么就进行合并
    注意返回的结果:如果可以合并,原来的dict就变为了数值
    """
    if (not is_tree(tree['left'])) and (not is_tree(tree['right'])):
        set_l, set_r = binary_split_data_set(data_test, tree['split_index'], tree['split_value'])
        error_no_merge = sum(power(set_l[:, -1] - tree['left'], 2)) + sum(power(set_r[:, -1] - tree['right'], 2))   # power(x, y)表示x的y次方
        mean_tree = (tree['left'] + tree['right']) / 2.0
        error_merge = sum(power(data_test[:, -1] - mean_tree, 2))
        if error_merge < error_no_merge:    # 如果合并的总方差<不合并的总方差,那么就进行合并
            print('合并')
            return mean_tree
        else:
            return tree
    else:
        return tree

def linear_solve(data_set):
    """
    desc:将数据集格式化成目标变量Y和自变量X,执行简单的线性回归,得到ws
    args:data_set:输入数据
    returns:ws:执行线性回归的回归系数,matrix_x:格式化自变量,matrix_y:格式化目标变量
    """
    m, n = shape(data_set)
    """
    产生一个关于1的矩阵
    """
    matrix_x = mat(ones((m, n)))
    matrix_y = mat(ones((m, 1)))
    matrix_x[:, 1 : n] = data_set[:, 0 : (n - 1)]
    matrix_y = data_set[:, -1]
    xTx = matrix_x.T * matrix_x # 转置矩阵*矩阵
    # 如果矩阵的逆不存在,会造成程序异常
    if linalg.det(xTx) == 0.0:
        raise NameError('该矩阵为奇异矩阵,不可逆,请尝试增加ops的第2个值')
    ws = xTx.I * (matrix_x.T * matrix_y)    # 最小二乘法求最优解:w0 * 1 + w1 * x1 = y
    return ws, matrix_x, matrix_y

def model_leaf(data_set):
    """
    desc:当数据不再需要切分的时候,生成叶节点的模型;得到模型的ws系数:f(x) = x0 + x1 * featrue1 + x3 * featrue2 ...;创建线性模型和返回系数
    args:data_set:输入数据集
    returns:调用函数linearSolve,返回得到的回归系数ws
    """
    ws, x, y = linear_solve(data_set)
    return ws

def model_error(data_set):
    """
    desc:在给定数据集上计算误差
    args:data_set:输入数据集
    returns:调用函数linear_solve,返回matrix_y_hat和matrix_y之间的平方误差
    """
    ws, matrix_x, matrix_y = linear_solve(data_set)
    matrix_y_hat = matrix_x * ws
    return sum(power(matrix_y - matrix_y_hat, 2))

def regress_tree_evaluate(model, data_input):
    """
    desc:对回归树进行预测(为了和函数model_tree_evaluate)
    args:model:指定模型,可选值为回归树模型或者模型树模型,这里为回归树;data_input:输入的测试数据
    returns:float(model):将输入的模型数据转换为浮点数返回
    """
    return float(model)

def model_tree_evaluate(model, data_input):
    """
    desc:对模型树进行预测(对输入数据进行格式化处理,在原数据矩阵上增加第0列,元素的值都是1,也就是增加偏移值,和之前的简单线性回归是一个套路,增加一个偏移量)
    args:model:输入模型,可选值为回归树模型或者模型树模型,这里为模型树模型;data_input:输入的测试数据
    returns:float(matrix_x * model):将测试数据乘以回归系数得到一个预测值,转化为浮点数返回
    """
    n = shape(data_input)[1]
    matrix_x = mat(ones((1, n + 1)))
    matrix_x[:, 1 : (n + 1)] = data_input
    return float(matrix_x * model)

def tree_forecast(tree, data_input, evaluate_model = regress_tree_evaluate):
    """
    desc:计算预测的结果;在给定树结构的情况下,对于单个数据点,该函数会给出一个预测值;evaluate_model是对叶节点进行预测的函数引用,指定树的类型,以便在叶节点上调用合适的模型;此函数自顶向下遍历整棵树,直到命中叶节点为止,一旦到达叶节点,它就会在输入数据上调用函数evaluate_model,该函数的默认值为regress_tree_evaluate;对特定模型的树进行预测,可以是回归树,也可以是模型树
    args:tree:已经训练好的树的模型;data_input:输入的测试数据;evaluate_model:预测的树的模型类型,可选值为regress_tree_evaluate(回归树)或model_tree_evaluate(模型树),默认为回归树
    returns:返回预测值
    """
    if not is_tree(tree):
        return evaluate_model(tree, data_input)
    if data_input[tree['split_index']] <= tree['split_value']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'], data_input, evaluate_model)
        else:
            return evaluate_model(tree['left'], data_input)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], data_input, evaluate_model)
        else:
            return evaluate_model(tree['right'], data_input)

def create_forecast(tree, data_test, evaluate_model = regress_tree_evaluate):
    """
    desc:调用tree_forecast,对特定模型的树进行预测,可以是回归树,也可以是模型树
    args:tree:已经训练好的树的模型;data_test:输入的测试数据;evaluate_model:预测的树的模型类型,可选值为regress_tree_evaluate(回归树)或model_tree_evaluate(模型树),默认为回归树
    returns:返回预测值矩阵
    """
    m = len(data_test)
    y_hat = mat(zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_forecast(tree, mat(data_test[i]), evaluate_model)
    return y_hat

if __name__ == '__main__':
    """
    回归树VS模型树VS线性回归
    """
    matrix_train = mat(load_data_set('./9.RegTrees/bikeSpeedVsIq_train.txt'))
    matrix_test = mat(load_data_set('./9.RegTrees/bikeSpeedVsIq_test.txt'))
    """
    回归树
    """
    regress_tree = create_tree(matrix_train, ops = (1, 20))
    print(regress_tree)
    y_hat_regress_tree = create_forecast(regress_tree, matrix_test[: , 0])
    print('-----------')
    print('回归树:',corrcoef(y_hat_regress_tree, matrix_test[:, 1], rowvar = 0)[0, 1])
    """
    模型树
    """
    model_tree = create_tree(matrix_train, model_leaf, model_error, ops = (1, 20))
    print(model_tree)
    y_hat_model_tree = create_forecast(model_tree, matrix_test[: , 0], model_tree_evaluate)
    print('-----------')
    print('模型树:',corrcoef(y_hat_model_tree, matrix_test[:, 1], rowvar = 0)[0, 1])
    """
    线性回归
    """
    ws, matrix_x, matrix_y = linear_solve(matrix_train)
    print(ws)
    m = len(matrix_test[:, 0])
    y_hat_linear_regress = mat(zeros((m, 1)))
    for i in range(shape(matrix_test)[0]):
        y_hat_linear_regress[i] = matrix_test[i, 0] * ws[1, 0] + ws[0, 0]
    print('线性回归:',corrcoef(y_hat_linear_regress, matrix_test[:, 1], rowvar = 0)[0, 1])
