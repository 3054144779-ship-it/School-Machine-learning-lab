from numpy import *
from bs4 import BeautifulSoup
from numpy import asmatrix as mat

def load_data_from_page(file_in, year, num_piece, price_origin, x_return, y_return):
    """
    读取HTML文件
    """
    fr = open(file_in, encoding = 'utf-8')
    soup = BeautifulSoup(fr.read(), "html5lib")
    fr.close()

    """
    根据HTML页面结构进行解析
    """
    i = 1
    current_row = soup.findAll('table', r = '%d' % i)
    while(len(current_row) != 0):
        title = current_row[0].findAll('a')[1].text
        title_lower = title.lower()
        """
        查找是否有全新标签
        """
        if (title_lower.find('new') > -1) or (title_lower.find('nisb') > -1):
            flag_new = 1
        else:
            flag_new = 0
        """
        查找是否已经标志出售,这里只收集已出售的数据
        """
        unicde_sold = current_row[0].findAll('td')[3].findAll('span')
        if len(unicde_sold) == 0:
            print('商品#%d没被出售' % i)
        else:
            """
            解析页面获取当前价格
            """
            price_sold = current_row[0].findAll('td')[4]
            price_str = price_sold.text
            price_str = price_str.replace('$', '')   # 去掉'$'
            price_str = price_str.replace(',', '')   # 去掉','
            if len(price_sold) > 1:
                price_str = price_str.replace('Free shipping', '')  # 去掉Free shipping
            price_selling = float(price_str)
            """
            去掉不完整的套装价格
            """
            if price_selling > 0.5 * price_origin:
                print("%d\t%d\t%d\t%f\t%f" % (year, num_piece, flag_new, price_origin, price_selling))
                x_return.append([year, num_piece, flag_new, price_origin])
                y_return.append(price_selling)
        i += 1
        current_row = soup.findAll('table', r = '%d' % i)

def ridge_regress(x_matrix, y_vector, lamda = 0.2):
    """
    desc:该函数实现了给定lamda下的岭回归求解;如果数据的特征比样本点还多,就不能再使用线性回归和局部线性回归了,因为计算(xTx)^(-1)会出现错误;如果特征比样本点还多(n>m),也就是说,输入数据的矩阵x不是满秩矩阵;菲满秩矩阵在求逆时会出现问题;为解决这一问题,使用岭回归,这是第一种缩减方法
    args:x_matrix:样本的特征数据;y_vector:每个样本对应的类别标签,即目标变量,实际值;lamda:引入的一个lamda值,使得矩阵非奇异
    returns:经过岭回归公式计算得到的回归系数
    """
    xTx = x_matrix.T * x_matrix
    denom = xTx + lamda * eye(shape(x_matrix)[1])
    if linalg.det(denom) == 0.0:
        print("该矩阵不可逆!")
        return []
    ws = denom.I * (x_matrix.T * y_vector)
    return ws

def ridge_test(x_matrix, y_vector):
    """
    desc:该函数用于在一组lamda上测试结果
    args:x_matrix:样本数据的特征;y_vector:样本数据的类别标签,即真实数据
    returns:w_matrix:将所有的回归系数输出到一个矩阵并返回
    """
    y_matrix = y_vector.T
    """
    mean(matrix, axis = 0),其中matrix为一个矩阵,axis为参数
    以m * n矩阵举例:
    axis不设置值,对m * n个数求均值,返回一个实数
    axis = 0:压缩行,对各列求均值,返回1 * n矩阵
    axis = 1:压缩列,对各行求均值,返回m * 1矩阵
    """
    y_mean = mean(y_matrix, 0)  # 计算y_matrix的均值
    y_matrix = y_matrix - y_mean    # y_matrix的所有特征减去均值
    x_means = mean(x_matrix, 0) # 标准化x,计算x_matrix平均值
    """
    var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue),其中a为一个矩阵,axis为参数
    以m * n矩阵为例:
    axis不设置值,对m*n个数求均值之后求矩阵中所有元素的方差,返回一个实数
    axis = 0:压缩行,对各列求均值,之后求各列中所有元素的方差,返回1 * n矩阵
    axis = 1:压缩列,对各行求均值,之后求各行中所有元素的方差,返回m * 1矩阵
    """
    x_var = var(x_matrix, 0)    # 然后计算x_matrix的方差
    x_handle = (x_matrix - x_means) / x_var # 所有特征都减去各自的均值并除以方差
    number_test = 30    # 可以在30个不同的lamda下调用函数ridge_regress
    w_matrix = zeros((number_test, shape(x_handle)[1])) # 创建number_test * x_handle的列数的全部数据为0的矩阵
    for i in range(number_test):
        ws = ridge_regress(x_handle, y_matrix, exp(i - 10)) # exp()返回e^x
        w_matrix[i, :] = ws.T
    return w_matrix

def cross_validation(x_array, y_array, num_valid = 10):
    """
    交叉验证测试岭回归
    """

    """
    获得数据点个数,x_array和y_array具有相同长度
    """
    m = len(y_array)
    index_list = list(range(m))
    error_matrix = zeros((num_valid, 30))
    """
    主循环,交叉验证循环
    """
    for i in range(num_valid):
        """
        随机拆分数据,将数据分为训练集(90%)和测试集(10%)
        """
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        """
        对数据进行混洗操作
        """
        random.shuffle(index_list)
        """
        切分训练集和测试集
        """
        for j in range(m):
            if j < m * 0.9:
                x_train.append(x_array[index_list[j]])
                y_train.append(y_array[index_list[j]])
            else:
                x_test.append(x_array[index_list[j]])
                y_test.append(y_array[index_list[j]])
        """
        获取回归系数矩阵
        """
        w_matrix = ridge_test(mat(x_train), mat(y_train))
        """
        循环遍历矩阵中的30组回归系数
        """
        for k in range(30):
            """
            读取训练集和数据集
            """
            x_test_matrix = mat(x_test)
            x_train_matrix = mat(x_train)
            """
            对数据进行标准化
            """
            x_train_mean = mean(x_train_matrix, 0)
            x_train_var = var(x_train_matrix, 0)
            x_test_matrix = (x_test_matrix - x_train_mean) / x_train_var
            """
            测试回归效果并存储
            """
            y_expect = x_test_matrix * mat(w_matrix[k, :]).T + mean(y_train)
            """
            计算误差
            """
            error_matrix[i, k] = ((y_expect.T.A - array(y_test)) ** 2).sum()
    """
    计算误差估计值的均值
    """
    mean_errors = mean(error_matrix, 0)
    mean_min = float(min(mean_errors))
    weights_best = w_matrix[nonzero(mean_errors == mean_min)]
    """
    不要使用标准化的数据,需要对数据进行还原来得到输出结果
    """
    x_matrix = mat(x_array)
    y_matrix = mat(y_array).T
    x_mean = mean(x_matrix, 0)
    x_var = var(x_matrix, 0)
    un_reg = weights_best / x_var
    """
    输出构建的模型
    """
    print("来自岭回归的最佳模型为", un_reg)
    print("常数项:", -1 * sum(multiply(x_mean , un_reg)) + mean(y_matrix))

def collect_data(x_return, y_return):
    load_data_from_page('./8.Regression/setHtml/lego8288.html', 2006, 800, 49.99, x_return, y_return)
    load_data_from_page('./8.Regression/setHtml/lego10030.html', 2002, 3096, 269.99, x_return, y_return)
    load_data_from_page('./8.Regression/setHtml/lego10179.html', 2007, 5195, 499.99, x_return, y_return)
    load_data_from_page('./8.Regression/setHtml/lego10181.html', 2007, 3428, 199.99, x_return, y_return)
    load_data_from_page('./8.Regression/setHtml/lego10189.html', 2008, 5922, 299.99, x_return, y_return)
    load_data_from_page('./8.Regression/setHtml/lego10196.html', 2009, 3263, 249.99, x_return, y_return)

def regress_test():
    x_list = []
    y_list = []
    collect_data(x_list, y_list)
    cross_validation(x_list, y_list, 10)

if __name__ == '__main__':
    regress_test()
