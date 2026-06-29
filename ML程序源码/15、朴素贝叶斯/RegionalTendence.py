from numpy.matrixlib.defmatrix import matrix
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

def calculate_most_frequence(list_vocabulary, text_full):
    """
    RSS源分类器及高频词去除函数
    """
    from operator import itemgetter
    dict_frequence = {}
    for token in list_vocabulary:
        dict_frequence[token] = text_full.count(token)
    frequence_sorted = sorted(dict_frequence.items(), key = itemgetter(1), reverse = True)
    return frequence_sorted[0:30]

def local_words(feed_1, feed_0):
    list_document = []
    list_class = []
    text_full = []
    print('0:', feed_0['entries'])
    l_0 = len(feed_0['entries'])
    print("len 0:", l_0)
    print('1:', feed_1['entries'])
    l_1 = len(feed_1['entries'])
    print("len 1:", l_1)
    len_min = min(l_0, l_1) # 找到两个中最小的一个
    for i in range(len_min):
        """
        类别1
        """
        list_word = text_parse(feed_1['entries'][i]['summary'])
        list_document.append(list_word)
        text_full.extend(list_word)
        list_class.append(1)
        """
        类别0
        """
        list_word = text_parse(feed_0['entries'][i]['summary'])
        list_document.append(list_word)
        text_full.extend(list_word)
        list_class.append(0)
    list_vocabulary = nb.create_element_list(list_document)
    """
    去掉高频词
    """
    words_top30 = calculate_most_frequence(list_vocabulary, text_full)
    for pair in words_top30:
        if pair[0] in list_vocabulary:
            list_vocabulary.remove(pair[0])
    """
    获取训练数据和测试数据
    """
    import random
    set_test = [int(number) for number in random.sample(range(2 * len_min), 20)]    # 生成随机取10个数,为了避免警告将每个数都转换为整型
    set_train = list(set(range(2 * len_min)) - set(set_test))   # 在原来的set_train中去掉这10个数
    """
    把这些训练集和测试集变成向量的形式
    """
    matrix_train = []
    class_train = []
    for index_document in set_train:
        matrix_train.append(nb.bag_elements2vector(list_vocabulary, list_document[index_document]))
        class_train.append(list_class[index_document])
        p0_v, p1_v, p_spam = nb.train_naive_bayes_correct(np.array(matrix_train), np.array(class_train))
    count_error = 0
    for index_document in set_test:
        vector_word = nb.bag_elements2vector(list_vocabulary, list_document[index_document])
        if nb.classify_naive_bayes(np.array(vector_word), p0_v, p1_v, p_spam) != list_class[index_document]:
            count_error += 1
    print('错误率:{}'.format(count_error /  float(len(set_test))))
    return list_vocabulary, p0_v, p1_v

def test_rss():
    import feedparser
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    list_vocabulary, p_sf, p_ny = local_words(ny, sf)

def get_top_words():
    import feedparser
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    list_vocabulary, p_sf, p_ny = local_words(ny, sf)
    ny_top = []
    sf_top = []
    for i in range(len(p_sf)):
        if p_sf[i] > -6.0:
            sf_top.append(list_vocabulary[i], p_sf[i])
        if p_ny[i] > -6.0:
            ny_top.append(list_vocabulary[i], p_ny[i])
    ny_sorted = sorted(ny_top, key = lambda pair: pair[1], reverse =  True)
    sf_sorted = sorted(sf_top, key = lambda pair: pair[1], reverse =  True)
    print('NY:')
    for item in ny_sorted:
        print(item[0])
    print("SF:")
    for item in sf_sorted:
        print(item[0])

if __name__ == '__main__':
    get_top_words()
