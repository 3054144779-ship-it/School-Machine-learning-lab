import numpy as np

def train_naive_bayes_original(train_matrix, train_category):
    """
    desc:朴素贝叶斯分类原版
    params:train_matrix:类型为ndarray,总的输入文件(eg:[[0,1,0,1],[],[]]);train_category:文件对应的类别分类(eg:[0,1,0]),列表长度应等于输入文本的长度
    returns:
    """
    train_document_number = len(train_matrix)
    words_number = len(train_matrix[0])
    """
    因为目标被标记为了1,所以只要把它们相加就可以得到目标有多少;目标文件的出现概率,即train_category中所有的1的个数,代表的就是多少个目标文件,与文件的总数相除就得到了目标文件的出现概率
    """
    pos_target = np.sum(train_category) / train_document_number
    """
    最小单元出现的次数(原版)
    """
    p0_number = np.zeros(words_number)
    p1_number = np.zeros(words_number)
    """
    整个数据集中最小单元出现的次数(原版是0)
    """
    p0_number_all = 0
    p1_number_all = 0
    for i in range(train_document_number):  # 遍历所有的文件
        if train_category[i] == 1:  # 如果是目标文件,就计算此目标文件中出现的目标最小单元的个数
            p1_number += train_matrix[i]
            p1_number_all += np.sum(train_matrix[i])
        else:
            p0_number += train_matrix[i]
            p0_number_all += np.sum(train_matrix[i])
    p1_vector = p1_number / p1_number_all
    p0_vector = p0_number / p0_number_all
    return p0_vector, p1_vector, pos_target

def train_naive_bayes_correct(train_matrix, train_category):
    """
    desc:朴素贝叶斯分类修正版(注意和原版的对比)
    params:train_matrix:类型为ndarray,总的输入文件(eg:[[0,1,0,1],[],[]]);train_category:文件对应的类别分类(eg:[0,1,0]),列表长度应等于输入文本的长度
    returns:
    """
    train_document_number = len(train_matrix)
    words_number = len(train_matrix[0])
    """
    因为目标被标记为了1,所以只要把它们相加就可以得到目标有多少;目标文件的出现概率,即train_category中所有的1的个数,代表的就是多少个目标文件,与文件的总数相除就得到了目标文件的出现概率
    """
    pos_target = np.sum(train_category) / train_document_number
    """
    最小单元出现的次数(修正版)(变为ones是为了防止数字过小溢出)
    """
    p0_number = np.ones(words_number)
    p1_number = np.ones(words_number)
    """
    整个数据集中最小单元出现的次数(修正版是2)
    """
    p0_number_all = 2.0
    p1_number_all = 2.0
    for i in range(train_document_number):  # 遍历所有的文件
        if train_category[i] == 1:  # 如果是目标文件,就计算此目标文件中出现的目标最小单元的个数
            p1_number += train_matrix[i]
            p1_number_all += np.sum(train_matrix[i])
        else:
            p0_number += train_matrix[i]
            p0_number_all += np.sum(train_matrix[i])
    """
    改成取log函数
    """
    p1_vector = np.log(p1_number / p1_number_all)
    p0_vector = np.log(p0_number / p0_number_all)
    return p0_vector, p1_vector, pos_target

def classify_naive_bayes(vector_classify, p0_vector, p1_vector, p_class1):
    """
    使用算法:将乘法转换为加法
    乘法:P(C|(F1F2...Fn)) = P((F1F2...Fn)|C) P(C) / P(F1F2...Fn)
    加法:P(F1|C)*P(F2|C)*...*P(Fn|C)*P(C)->log(P(F1|C))+log(P(F2|C))+...+log(P(Fn|C))+log(P(C))
    params:vector_classify:带测数据(eg:[0,1,1,1,1,...]),即要分类的向量;p0_vector:类别0,即一般文件的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0)),...]列表;p1_vector:类别1,即目标文件的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1)),...]列表;p_class1:类别1,目标文件的出现概率
    returns:类别1/0
    """
    """
    计算公式:log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    使用NumPy数组来计算两个向量相乘的结果,这里的相乘是指对应元素相乘,即先将两个向量中的第一个元素相乘,然后将第2个元素相乘,...,以此类推
    本人的理解是:这里的vector_classify * p1_vector的意思就是将每个词与其对应的概率相关联起来
    可以理解为:1.文件在单元表中的条件下,文件是类别0的概率;也可以理解为:2.在整个空间下,文件既在单元表中又是类别0的概率
    """
    p1 = np.sum(vector_classify * p1_vector) + np.log(p_class1)
    p0 = np.sum(vector_classify * p0_vector) + np.log(1 - p_class1)
    return (1 if (p1 > p0) else 0)

def create_element_list(data_set):
    """
    desc:获取所有元素的集合
    params:data_set:数据集
    return:所有元素的集合(即不含重复元素的元素列表)
    """
    element_set = set() # 创建空集
    for item in data_set:
        element_set = element_set | set(item)   # 求两个集合的并集
    return list(element_set)

def set_of_element2vector(element_list, input_set):
    """
    desc:遍历查看该元素是否出现,出现该元素则将该元素置1
    params:element_list:所有元素集合列表;input_set:输入数据集
    returns:匹配列表(eg:[0,1,0,1,...]),其中1与0表示列表中的元素是否出现在输入的数据集中
    """
    result = [0] * len(element_list)    # 创建一个与元素列表登场的向量,并将其元素都设置为0
    """
    遍历文件中的所有元素,如果出现了元素列表中的元素,则将输出的文件向量中的对应值设为1
    """
    for element in input_set:
        if element in element_list:
            result[element_list.index(element)] = 1
    return result

def bag_elements2vector(element_list, input_set):
    """
    注意和原先的做对比
    """
    result = [0] * len(element_list)
    for element in input_set:
        if element in element_list:
            result[element_list.index(element)] += 1
        else:
            print("元素:{}不在元素列表中".format(element))
    return result
