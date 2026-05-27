from numpy import *

def load_data_set(name_file):   # 从文本中构建矩阵,加载文本文件,然后处理
    """
    通用函数,用来解析以tab键分隔的floats(浮点数)
    """
    set_data = []
    fr = open(name_file)
    str_all = fr.readlines()
    fr.close()
    for str_line in str_all:
        list_float = []
        str_current = str_line.strip().split('\t')  # 映射所有的元素为float(浮点数)类型
        list_float = list(map(float, str_current))
        # for str_float in str_current:
        #     list_float.append(float(str_float))
        set_data.append(list_float)
    print('set_data:', set_data)
    return set_data

def distance_Euclidean(vector_A, vector_B):
    """
    计算两个向量的欧式距离(可根据场景选择)
    """
    return sqrt(sum(power(vector_A - vector_B, 2)))

def random_centroid(matrix_data, k):
    """
    为给定数据集构建一个包含k个随机质心的集合,随机质心必须要在整个数据集的边界之内,这可以通过找到数据集每一维的最小和最大值来完成;然后生成0~1.0之间的随机数并通过取值范围和最小值,以便确保随机点在数据的边界之内
    """
    n = shape(matrix_data)[1]   # 列的数量
    centroids = mat(zeros((k, n)))   # 创建k个质心矩阵
    """
    创建随机簇质心,并且在每一维的边界内
    """
    for j in range(n):
        j_min = min(matrix_data[:, j])  # 最小值
        j_range = float(max(matrix_data[:, j]) - j_min) # 范围 = 最大值 - 最小值
        centroids[:, j] = mat(j_min + j_range * random.rand(k, 1))   # 随机生成
    return centroids

def k_means(matrix_data, k, distance_means = distance_Euclidean,create_centroid = random_centroid):
    """
    desc:该算法会创建k个质心,然后将每个点分配到最近的质心,再重新计算质心;这个过程重复数次,直到数据点的簇分配结果不再改变位置;运行结果(多次运行结果可能会不一样,可以试试,原因为随机质心的影响,但总的结果是对的,因为数据足够相似,也可能会陷入局部最小值)
    """
    m = shape(matrix_data)[0]   # 行数
    cluster_assment = mat(zeros((m, 2)))    # 创建一个与matrix_data行数一样,但是有两列的矩阵,用来保存簇分配结果
    centroids = create_centroid(matrix_data, k) # 创建质心,随机k个质心
    b_cluster_changed = True
    while b_cluster_changed:
        b_cluster_changed = False
        """
        循环每一个数据点并分配到最近的质心中去
        """
        for i in range(m):
            distance_min = inf
            index_min = -1
            for j in range(k):
                distance_j_i = distance_means(centroids[j, :], matrix_data[i, :])   # 计算数据点到质心的距离
                """
                如果距离比distance_min(最小距离)还小,更新distance_min(最小距离)和最小质心的索引
                """
                if distance_j_i < distance_min:
                    distance_min = distance_j_i
                    index_min = j
            if cluster_assment[i, 0] != index_min:  # 簇分配结果改变
                b_cluster_changed = True    # 簇改变
                cluster_assment[i, :] = index_min, distance_min ** 2    # 更新簇分配结果为最小质心的索引,最小距离distance_min的平方
        print(centroids)
        """
        更新质心
        """
        for centroid in range(k):
            points_in_cluster = matrix_data[nonzero(cluster_assment[:, 0].A == centroid)[0]]    # 获取该簇中的所有点
            centroids[centroid, :] = mean(points_in_cluster, axis = 0)  # 将质心修改为簇中所有点的平均值,mean就是求平均值的
    return centroids, cluster_assment

def binary_k_means(matrix_data, k, distance_means = distance_Euclidean):
    m = shape(matrix_data)[0]
    cluster_assment = mat(zeros((m, 2)))    # 保存每个数据点的簇分配结果和平方误差
    centroid_0 = mean(matrix_data, axis = 0).tolist()[0]    # 质心初始化为所有数据点的均值
    list_centroids = [centroid_0]   # 初始化只有1个质心的list
    """
    计算所有数据点到初始质心的距离平方误差
    """
    for j in range(m):
        cluster_assment[j, 1] = distance_means(mat(centroid_0), matrix_data[j, :]) ** 2
    while len(list_centroids) < k:  # 当质心数量小于k时
        lowest_SSE = inf
        for i in range(len(list_centroids)):    # 对每一个质心
            points_in_current_cluster = matrix_data[nonzero(cluster_assment[:, 0].A == i)[0], :]    # 获取当前簇i下的所有数据点
            matrix_centroids, cluster_assment_split = k_means(points_in_current_cluster, 2, distance_means) # 将当前簇i进行二分kMeans处理
            split_SSE = sum(cluster_assment_split[:, 1])    # 将二分kMeans结果中的平方和的距离进行求和
            not_split_SSE = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])   # 将未参与二分kMeans分配结果中的平方和的距离进行求和
            print('split_SSE:{};not_split_SSE:{}'.format(split_SSE, not_split_SSE))
            if (split_SSE + not_split_SSE) < lowest_SSE:
                centroid_best_to_split = i
                centroids_new_best = matrix_centroids
                cluster_assment_best = cluster_assment_split.copy()
                lowest_SSE = split_SSE + not_split_SSE
        """
        找出最好的簇分配结果
        """
        cluster_assment_best[nonzero(cluster_assment_best[:, 0].A == 1)[0], 0] = len(list_centroids)    # 调用二分kMeans的结果,默认簇是0,1;当然也可以改成其它的数字
        cluster_assment_best[nonzero(cluster_assment_best[:, 0].A == 0)[0], 0] = centroid_best_to_split # 更新为最佳质心
        print('centroid_best_to_split:', centroid_best_to_split)
        print('cluster_assment_best的长度:', len(cluster_assment_best))
        """
        更新质心列表
        """
        list_centroids[centroid_best_to_split] = centroids_new_best[0, :].tolist()[0]   # 更新原质心list中的第i个质心为使用二分kMeans后centroids_new_best的第一个质心
        list_centroids.append(centroids_new_best[1, :].tolist()[0]) # 添加centroids_new_best的第二个质心
        cluster_assment[nonzero(cluster_assment[:, 0].A == centroid_best_to_split)[0], :] = cluster_assment_best    # 重新分配最好簇下的数据(质心)以及SSE
    return mat(list_centroids), cluster_assment

def basic_functions_test():
    matrix_data = mat(load_data_set('./10.KMeans/testSet.txt')) # 加载测试数据集
    print('matrix_data:', matrix_data)
    """
    首先,先看一下矩阵中的最大值与最小值
    """
    print('min(matrix_data[:, 0]) = ', min(matrix_data[:, 0]))
    print('min(matrix_data[:, 1]) = ', min(matrix_data[:, 1]))
    print('max(matrix_data[:, 1]) = ', max(matrix_data[:, 1]))
    print('max(matrix_data[:, 0]) = ', max(matrix_data[:, 0]))
    """
    然后看看函数random_centroid能否生成min到max之间的值
    """
    print('random_centroid(matrix_data, 2) = ', random_centroid(matrix_data, 2))
    """
    最后测试一下距离计算方法
    """
    print('distance_Euclidean(matrix_data[0], matrix_data[1]) = ', distance_Euclidean(matrix_data[0], matrix_data[1]))

def k_means_test():
    matrix_data = mat(load_data_set('./10.KMeans/testSet.txt')) # 加载测试数据集
    """
    该算法会创建k个质心,然后将每个点分配到最近的质心,再重新计算质心;这个过程重复数次,知道数据点的簇分配结果不再改变位置;运行结果(多次运行结果可能会不一样,可以试试,原因为随机质心的影响,但总的结果是对的,因为数据足够相似)
    """
    centroids, cluster_assment = k_means(matrix_data, 4)
    print('质心:', centroids)

def binary_k_means_test():
    matrix_data = mat(load_data_set('./10.KMeans/testSet2.txt')) # 加载测试数据集
    list_centroids, cluster_assments = binary_k_means(matrix_data, 3)
    print('质心序列:',list_centroids)

if __name__ == '__main__':
    basic_functions_test()
    k_means_test()
    binary_k_means_test()
