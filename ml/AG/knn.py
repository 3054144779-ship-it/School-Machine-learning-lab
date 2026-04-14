from BaseAlgorithm import BaseAlgorithm

class KNN(BaseAlgorithm) :
    def __init__(self, k=20) :
        self.k = k
    
    def fit(self, data, labels) :
        self.train_data = data
        self.train_labels = labels
    
    def predict(self, input_data):
        # 计算差值矩阵 
        diffMat = self.train_data - input_data
        
        # 平方并求和 (axis=1 表示按行求和)
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        
        # 开方得到最终距离
        distances = sqDistances**0.5
        
        # 获取排序后的下标 
        sortedDistIndices = distances.argsort()
        
        # 统计前 K 个邻居的标签
        classCount = {}
        for i in range(self.k):
            voteIlabel = self.train_labels[sortedDistIndices[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        
        # 排序选出得票最高的类别
        sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
        
        # 返回最高类别
        return sortedClassCount[0][0]

    





