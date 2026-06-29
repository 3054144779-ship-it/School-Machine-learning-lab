import os
import pandas as pd

class Data :
    def __init__(self, input_path, output_path):
        self.input_path = input_path     # 数据集所在的路径 
        self.output_path = output_path   # 数据输出到的路径

    
    '''
    清洗数据

    :param range_limit: 需要清洗的数据和数据范围
    格式:
    range_limit = {
        "test1" : [left, right],
        "test2" : [left, right]
    }

    :param feature_limit: 需要进行独热编码的类别特征列表
    :return 返回清洗后的 DataFrame 供下一步特征选择使用
    '''
    def data_show(self, range_limit, feature_limit=[]) :
        # 读取 excel 表格的数据集
        df = pd.read_excel(self.input_path)
        '''
        orient="records": 输出 [{}, {}, {}] 数组格式
        force_ascii=False: 中文不乱码
        indent=2: 格式化, 方便阅读, 去掉则压缩一行
        '''

        # 寻找缺失值并补充
        for col in range_limit.keys() : # 取出每一列的名字
            # 当前列没有缺失值则跳过
            if df[col].isnull().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                # 如果是数字的话, 每列以均值填充, 直接重新赋值
                df[col] = df[col].fillna(df[col].mean())
            else :
                # 如果是字符串, 以众数字符串填充
                df[col] = df[col].fillna(df[col].mode()[0])

        # 按照范围需要过滤的列
        for key, (min_value, max_value) in range_limit.items() :
            df = df[df[key].between(min_value, max_value)] # between(左,右) 是闭区间，包含左右边界
        
        # 生成类别特征（仅在 feature_limit 非空时执行）
        if feature_limit:
            dummies = pd.get_dummies(df[feature_limit], prefix=feature_limit)
            '''
            concat 拼接函数，用来合并多个 DataFrame 表格
            传入一个列表，代表要合并的两张表：
            df: 清洗后的完整表格
            dummies: pd.get_dummies() 生成的独热编码表，只有一堆 0、1 的新特征列
            axis=1 → 按列合并
            axis=0 → 按行合并
            '''
            df = pd.concat([df, dummies], axis=1)
            df.drop(feature_limit, axis=1, inplace=True)

        # 变成 json 文件，每行一条对象
        df.to_json(self.output_path, orient="records", force_ascii=False, indent=2)

        # 返回清洗后的 DataFrame 供下一步特征选择使用
        return df
    

    '''
    特征选择

    :param df: 清洗过后的 DataFrame 数据
    :param target_col: 目标变量的列名
    :param threshold: 相关性系数阈值(绝对值)，低于此值的特征将被过滤

    :return 保留下来的特征列和目标列，返回最终的数据集
    '''

    def feature_choose(self, df, target_col, threshold=0.1) :
        # 只对数值型列计算相关性 (One-Hot后的特征也是数值型)
        numeric_df = df.select_dtypes(include=['int64', 'float64', 'uint8', 'bool'])
        
        # 计算所有特征与目标变量的 Pearson 相关系数
        correlations = numeric_df.corr(method='pearson')[target_col]  # corr 计算整张表的相关矩阵，method设置为计算相关系数
        
        # 取绝对值以评估相关性强度 (无论是正相关还是负相关)
        abs_correlations = correlations.abs()
        
        # 筛选出相关系数绝对值大于阈值的特征
        selected_features = abs_correlations[abs_correlations > threshold].index.tolist()
        
        # 将目标列本身从特征列表中移除
        if target_col in selected_features:
            selected_features.remove(target_col)
            
        print(f"--- 特征选择完成 ---")
        print(f"目标变量: {target_col}")
        print(f"保留的特征 ({len(selected_features)}个): {selected_features}")
        
        # 组合保留下来的特征列和目标列，返回最终的数据集
        final_df = df[selected_features + [target_col]]
        return final_df
