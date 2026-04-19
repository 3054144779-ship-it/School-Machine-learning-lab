from numpy import *
# import operator
from os import listdir
# from collections import Counter

def img2vector(file_name):
    """
    目的:将图像数据转换为向量
    params:file_name:图片文件(输入数据的图片格式为32*32)
    return:一维矩阵(向量)
    该函数将图像转换为向量:该函数创建一个1*1024的Numpy数组,然后打开指定文件,循环读出文件的前32行,并将每行的头32个字符值存储在Numpy数组中,最后返回数组
    """
    return_vector = zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line_str[j])
    fr.close()
    return return_vector

def classify_knn(x_in,data_set,labels,k):
    """
    params:
        x_in:要分类的输入向量
        data_set:输入的训练样本集
        labels:标签向量(输入的训练样本集对应的标签向量)
        k:选择最近邻的数目
    notes:
        labels元素数目=data_set行数
        程序使用欧拉距离公式
    """
    # 1.计算距离
    data_set_size = data_set.shape[0]   # data_set_size:数据集data_set的第一个维度(即行数)
    diff_matrix = tile(x_in,(data_set_size,1))-data_set  # 使用tile生成和训练样本对应的矩阵,并与训练样本求差
    """
    欧氏距离:点到点之间的距离
    """
    square_diff_matrix = diff_matrix ** 2   # 每个元素取平方构成矩阵square_diff_matrix
    square_distance_vector = square_diff_matrix.sum(axis=1)   # 将矩阵square_diff_matrix的每一行中所含的各个元素相加,构成一个square_diff_matrix.shape[0](亦即data_set_size)*1的向量square_distance_vector
    distances_vector = square_distance_vector ** 0.5    # 将向量square_distance_vector中的每个元素开方后得到向量distances_vector
    """
    y=x.argsort():将x中的元素升序排列,提取其对应的索引,然后输出到y
    eg:x=array([1,4,3,-1,6,9]),y=x.argsort()=array([3,0,2,1,4,5]);x[3]=-1(min),y[0]=3;x[5]=9(max),y[5]=5
    """
    index_ascend_sorted_vector = distances_vector.argsort() # 根据距离从小到大排序,返回对应的索引,构成相应的向量index_ascend_sorted_vector
    # 2.选择距离最小的k个点
    class_count = {}
    for i in range(k):
        # print(index_ascend_sorted_vector[i])
        vote_label = labels[index_ascend_sorted_vector[i]]  # 找到该样本的类型
        """
        字典的get方法:
        dict.get(k,d),其中:get相当于一条if...else...语句;参数k在字典中,字典将返回dict[k](即k对应的value值);如果参数k不在字典中,则返回参数d
        """
        class_count[vote_label] = class_count.get(vote_label, 0) + 1    # 在字典中将该类型对应的计数量+1
    # 3.排序并返回出现次数最多的那个类型
    key_max_class_cnt = max(class_count, key=class_count.get)   # 利用函数max直接返回字典中value最大的键
    return key_max_class_cnt

def number_hand_writing_test():
    # 1.导入数据
    labels_hand_writing = []
    file_list_training = listdir('./2.KNN/trainingDigits')  #导入训练集
    m = len(file_list_training)
    matrix_training = zeros((m, 1024))
    # labels_hand_writing存储0~9对应的索引位置,matrix_training存放每个位置对应的图片向量
    for i in range(m):
        str_file_name = file_list_training[i]
        str_file = str_file_name.split('.')[0]  # 以'.'分割文件名对应的字符串,并提取其中第一个字符串(文件名)
        str_class_name = int(str_file.split('_')[0])    #将文件名用'_'分割,并提取其中第一个字符串,之后转为数字
        labels_hand_writing.append(str_class_name)
        matrix_training[i, :] = img2vector('./2.KNN/trainingDigits/%s' % str_file_name)  #读取对应文件,并将32*32的矩阵转换为1*1024的矩阵(向量)

    # 2.导入测试数据
    list_test_file = listdir('./2.KNN/testDigits')  #导入测试数据集
    count_error = 0 # 初始化错误计数量
    n_test = len(list_test_file)
    for i in range(n_test):
        str_file_name = list_test_file[i]
        str_file = str_file_name.split('.')[0]
        str_class_number = int(str_file.split('_')[0])
        vector_test = img2vector('./2.KNN/testDigits/%s' % str_file_name)
        result_classify = classify_knn(vector_test, matrix_training, labels_hand_writing, 3)
        print("分类结果:%d,真实结果:%d" % (result_classify, str_class_number))
        if result_classify != str_class_number:
            count_error += 1
    print('总错误数:%d,错误率:%f' % (count_error, count_error / float(n_test)))

if __name__ == '__main__':
    number_hand_writing_test()