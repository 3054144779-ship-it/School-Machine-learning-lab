import NaiveBayes as nb
import numpy as np

def text_parse(str_upper):
    """
    desc:做词划分
    params:str_upper:某个被拼接后的字符串
    returns:全都是小写的单词列表,去掉少于2个字符的字符串
    """
    import re
    token_list = re.split(r'\W+', str_upper)    # 推荐用'\W+'代替'\W*',因为'\W*'会匹配空模式,在python3.5+之后就会出现问题
    if len(token_list) == 0:
        print(token_list)
    return [tk.lower() for tk in token_list if len(tk) > 2]

def spam_test():
    """
    desc:对贝叶斯垃圾邮件分类器进行自动化处理
    params:无
    returns:无
    """
    list_document = []
    list_class = []
    text_full = []
    for i in range(1, 26):
        """
        添加垃圾邮件信息
        使用try,except来做的原因:
        因为数据集中有几个文件编码格式为windows 1252(spam:17.txt,ham:6.txt,...)
        其实还可以按如下的方法去做:
        import os
        检查os.system('file {}.txt'.format(i)),看下返回的是什么
        如果正常能读返回的都是:ASCII text
        对于异常需要处理的返回的都是:Non-ISO extended-ASCII text,with very long lines
        """
        try:
            words = text_parse(open('./4.NaiveBayes/email/spam/{}.txt'.format(i)).read())
        except:
            words = text_parse(open('./4.NaiveBayes/email/spam/{}.txt'.format(i), encoding = 'Windows 1252').read())
        list_document.append(words)
        text_full.extend(words)
        list_class.append(1)
        """
        添加非垃圾邮件
        """
        try:
            words = text_parse(open('./4.NaiveBayes/email/ham/{}.txt'.format(i)).read())
        except:
            words = text_parse(open('./4.NaiveBayes/email/ham/{}.txt'.format(i), encoding = 'Windows 1252').read())
        list_document.append(words)
        text_full.extend(words)
        list_class.append(0)
    list_vocabulary = nb.create_element_list(list_document) # 创建词汇表
    import random
    set_test = [int(number) for number in random.sample(range(50), 10)] # 生成随机取10个数,为了避免警告将每个数都转换为整型
    set_train = list(set(range(50)) - set(set_test))   # 在原来的set_train中去掉这10个数
    matrix_train = []
    class_train = []
    for index_document in set_train:
        matrix_train.append(nb.set_of_element2vector(list_vocabulary, list_document[index_document]))
        class_train.append(list_class[index_document])
        p0_v, p1_v, p_spam = nb.train_naive_bayes_correct(np.array(matrix_train), np.array(class_train))
    """
    开始测试
    """
    count_error = 0
    for index_document in set_test:
        vector_word = nb.set_of_element2vector(list_vocabulary, list_document[index_document])
        if nb.classify_naive_bayes(np.array(vector_word), p0_v, p1_v, p_spam) != list_class[index_document]:
            count_error += 1
    print("错误率:{}".format(count_error/float(len(set_test))))

if __name__ == '__main__':
    spam_test()
