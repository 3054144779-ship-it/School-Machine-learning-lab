import NaiveBayes as nb
import numpy as np

def load_data_set():
    """
    desc:创建数据集(都是假的)
    return:posting_list:单词列表;class_vector:所属类别
    """
    posting_list = [
                    ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                    ]
    class_vector = [0, 1, 0, 1, 0, 1]  # 1:侮辱性的文字, 0:非侮辱性的文字
    return posting_list, class_vector

def test_abusive_remarks():
    list_post, list_classes = load_data_set()   # 加载数据集
    list_vocabulary = nb.create_element_list(list_post) # 创建单词集合
    """
    计算单词是否出现并创建数据矩阵
    """
    train_matrix = []
    for input_post in list_post:
        train_matrix.append(nb.set_of_element2vector(list_vocabulary, input_post))  # 返回m * len(list_vocabulary)的矩阵,记录的都是0,1信息;其实就是那个东西的句子向量(就是数据集里面的每一行,也不算句子)
    p0_v, p1_v, pos_abusive = nb.train_naive_bayes_correct(np.array(train_matrix), np.array(list_classes))  # 训练数据
    """
    测试数据
    """
    test_1 = ['love', 'my', 'dalmation']
    test_1_document = np.array(nb.set_of_element2vector(list_vocabulary, test_1))
    print("结果为{}".format(nb.classify_naive_bayes(test_1_document, p0_v, p1_v, pos_abusive)))
    test_2 = ['stupid', 'garbage']
    test_2_document = np.array(nb.set_of_element2vector(list_vocabulary, test_2))
    print("结果为{}".format(nb.classify_naive_bayes(test_2_document, p0_v, p1_v, pos_abusive)))

if __name__ == '__main__':
    test_abusive_remarks()