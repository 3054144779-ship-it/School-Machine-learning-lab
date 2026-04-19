import operator
from math import log, pi
import DecisionTreePlot as dt_plt
from collections import Counter

def create_data_set():
    """
    desc:生成基础数据集
    args:无
    returns:返回数据集和对应的标签
    """
    data_set = [
                [1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']
                ]
    # labels = ["出水能否存活", "有无脚蹼"]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

def calculate_shannon_entropy(data_set):
    number_entries = len(data_set)  # 求list的长度,表示计算参与训练的数据量
    label_counts = {}   # 计算分类标签label出现的次数
    for feature_vector in data_set: # 唯一元素的数量即发生率
        current_label = feature_vector[-1] # 将当前实例的标签存储,即每行数据的最后一个数据代表的是标签
        """
        为所有可能的分类创建字典,如果当前的键值不存在,则扩展字典并将当前键值加入字典;每个键值都记录了当前类别出现的次数
        """
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    """
    对于label标签的占比Z,求出label标签的香农熵
    """
    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / number_entries    # 使用所有类标签的发生概率计算类别出现的概率
        shannon_entropy -= prob * log(prob, 2)  # 计算香农熵,以2为底求对数
    return shannon_entropy

def split_data_set(data_set, index, value):
    """
    desc:通过遍历数据集data_set,求出index对应的colnum列的值为value的行;即依据列index进行分类,如果index列的数据等于value的时候,就要将index划分到所创建的新的数据集中
    args:data_set:数据集(待划分的数据集);index:表示每一行的index列(划分数据集的特征);value:index列对应的value值(需要返回的特征的值)
    returns:index列为value的数据集(该数据集需要排除index列)
    """
    return_data_set = []
    for feature_vector in data_set:
        """
        index列为value的数据集(该数据集需要排除index列)
        判断index列的值是否为value
        """
        if feature_vector[index] == value:
            reduce_feature_vector = feature_vector[:index]    # 切掉用于分割的索引;[:index]表示前index行
            """
            extend和append的区别:
            x.append(object)向列表添加一个对象object;使用append的时候,是将object看作一个对象,整体打包添加到x对象中
            x.extend(sequence)把一个序列sequence的内容添加到列表中(跟+=在列表运用类似);使用extend的时候,是将sequence看作一个序列,将这个序列和x序列合并,并放在其后面
            """
            reduce_feature_vector.extend(feature_vector[(index + 1):])    # [(index + 1):]表示从跳过index的index + 1行,取接下来的数据
            return_data_set.append(reduce_feature_vector)  # 收集结果值,index列为value的行(该行需要排除index列)
    return return_data_set

def choose_best_feature_to_split(data_set):
    """
    desc:选择最好的特征
    args:data_set:数据集
    return:feature_best:最优的特征列
    """
    number_features = len(data_set[0]) - 1  # 求第一行有多少列的特征,最后一列是label列
    base_entropy = calculate_shannon_entropy(data_set)  # 数据集的原始信息熵
    best_info_gain, best_feature = 0.0, -1  # 最优的信息增益值和最优的特征编号
    """
    对所有特征进行迭代
    """
    for i in range(number_features):
        feature_list = [example[i] for example in data_set] # 获取对应的特征下的所有数据
        unique_values = set(feature_list)   # 获取剔重后的集合,使用set对list数据进行去重
        new_entropy = 0.0   # 创建一个临时的信息熵
        """
        遍历某一列的值集合,计算该列的信息熵
        遍历当前特征中的所有唯一属性值,对每个唯一属性值划分一次数据集,计算数据集的新熵值,并对所有唯一特征值得到的熵求和
        """
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set)) # 计算概率
            new_entropy += prob * calculate_shannon_entropy(sub_data_set)   # 计算信息熵
        info_gain = base_entropy - new_entropy  # gain[信息增益]:划分数据集前后的信息变化,获取信息熵最大的值;信息增益是熵的减少或者数据无序度的减少;最后,比较所有特征中的信息增益,返回最好特征划分的索引值
        print("信息增益:", info_gain, "最佳特征:", i, "原始信息熵:", base_entropy, "新信息熵:", new_entropy)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    """
    desc:选择出现次数最多的一个结果
    args:class_list:label列的集合
    return:best_feature:最优的特征列
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)  #倒序排列class_count得到一个字典集合,然后取出第一个就是结果(yes/no),即出现次数最多的结果
    return sorted_class_count[0][0]

def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]   #如果数据集的最后一列的第一个值出现的次数 = 整个集合的数量,也就是说只有一个类别,就直接返回结果就行
    """
    第一个停止条件:所有的类标签完全相同,则直接返回该类标签
    函数count()是统计()中的值在list中出现的次数
    """
    if class_list.count(class_list[0]) == len(class_list):  # 如果数据集只有1列,那么最初出现label次数最多的一类,作为结果
        return class_list[0]
    """
    第二个停止条件:使用完了所有特征,仍然不能将数据集画风称仅包含唯一类别的分组
    """
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_best_feature_to_split(data_set)   # 选择最优的列,得到最优列对应的label含义
    best_feature_label = labels[best_feature]   # 获取label的名称
    my_tree = {best_feature_label:{}}   # 初始化my_tree
    del(labels[best_feature])   # labels列表是可变对象,在python函数中作为参数时传址引用,能够被全局修改;所以这行代码导致函数外的同名变量被删除了元素,造成例句无法执行
    feature_values = [example[best_feature] for example in data_set]    # 取出最优列,然后它的分支做分类
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]  # 求出剩余的标签label
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)
    return my_tree

def classify(input_tree, feature_labels, test_vector):
    """
    desc:给输入的节点,进行分类
    args:input_tree:决策树模型;feature_labels:特征标签对应的名称;test_vector:测试输入的数据
    returns:class_label:分类的结果值,需要映射label才能知道名称
    """
    first_str = list(input_tree.keys())[0]    # 获取树的根节点对应的key值
    second_dict = input_tree[first_str] # 通过key得到根节点对应的值
    feature_index = feature_labels.index(first_str) # 判断根节点名称获取根节点在label中的先后顺序,这样就知道输入的test_vector怎么开始对照树来做分类
    """
    测试数据,找到根节点对应的标签位置,也就知道从输入的数据的第几位来开始分类
    """
    key = test_vector[feature_index]
    value_of_feature = second_dict[key]
    print('+++', first_str, 'xxx', second_dict, '---', key, '>>>', value_of_feature)
    """
    判断分支是否结束:判断value_of_feature是否为dict类型
    """
    if isinstance(value_of_feature, dict):
        class_label = classify(value_of_feature, feature_labels, test_vector)
    else:
        class_label = value_of_feature
    return class_label

def store_tree(input_tree, file_name):
    """
    desc:将之前训练好的决策树模型存储起来,使用pickle模块
    args:input_tree:以前训练好的决策树模型;file_name:要存储的名称
    returns:无
    """
    import pickle
    with open(file_name, 'wb') as fw:
        pickle.dump(input_tree, fw)

def grab_tree(file_name):
    """
    desc:将之前存储的决策树模型使用pickle模块还原出来
    args:file_name:之前存储决策树模型的文件名
    returns:pickle.load(fr):将之前存储的决策树模型还原出来
    """
    import pickle
    with open(file_name, 'rb') as fr:
        return pickle.load(fr)

def fish_test():
    """
    desc:对动物是否是鱼类分类的测试函数,并将结果使用matplotlib画出来
    args:无
    returns:无
    """
    data_set, labels = create_data_set()    # 创建数据和结果标签
    import copy
    my_tree = create_tree(data_set, copy.deepcopy(labels))
    print(my_tree)
    print(classify(my_tree, labels, [1, 1]))    # [1, 1]表示要取的分支上的节点位置,对应的结果值
    dt_plt.create_plot(my_tree) # 画图可视化展示

if __name__ == '__main__':
    fish_test()
