import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data():
    '''
    用于加载并合并多个CSV文件中的数据, 并为合并后的数据添加列名。
    '''
    dfs = []
    for i in range(1,5):
        path = 'archive/UNSW-NB15_{}.csv'
        dfs.append(
            pd.read_csv(
                path.format(i), 
                header=None
            )
        )
    combined_data = pd.concat(dfs).reset_index(drop=True)
    
    dataset_columns = pd.read_csv('archive/NUSW-NB15_features.csv', encoding='ISO-8859-1')
    combined_data.columns = dataset_columns['Name']
    return combined_data

def clean_data(data):
    '''
    清洗数据集中的部分列，包括：
    'attack_cat' 列的缺失值用 'normal' 填充，并统一格式为小写、去除空格。
    'ct_flw_http_mthd' 和 'is_ftp_login' 列的缺失值用 0 填充。
    将 'is_ftp_login' 中大于 1 的值截断为 1（二元化处理）。
    将 'attack_cat' 中的 'backdoors' 统一为 'backdoor'。
    将 'service' 列中的 '-' 替换为 'None'。
    将 'ct_ftp_cmd' 列中的空格替换为 0 并转换为整型。
    
    data: 输入的数据集
    '''
    # 填充缺失值
    data['attack_cat'] = data['attack_cat'].fillna(value='normal').apply(lambda x: x.strip().lower())
    data['attack_cat'] = data['attack_cat'].replace('backdoors','backdoor', regex=True)
    data['ct_flw_http_mthd'] = data['ct_flw_http_mthd'].fillna(value=0)
    data['is_ftp_login'] = data['is_ftp_login'].fillna(value=0)
    data['is_ftp_login'] = np.where(data['is_ftp_login']>1, 1, data['is_ftp_login'])
    
    # 处理service列
    data['service'] = data['service'].apply(lambda x:"None" if x=='-' else x)
    
    # 处理ct_ftp_cmd列
    data['ct_ftp_cmd'] = data['ct_ftp_cmd'].replace(to_replace=' ', value=0).astype(int)
    
    return data

def prepare_data(data):
    '''
    预处理数据，包括：
    删除无用列（srcip、sport、dstip、dsport、Label）
    将数据集按8:2划分为训练集和测试集
    再将训练集按8:2划分为训练集和验证集
    分离特征与标签（attack_cat为标签）
    返回训练集、测试集和验证集的特征与标签
    
    data: 输入的数据集
    
    x_train: 训练集的特征数据，去除了标签列
    y_train: 训练集的标签数据，仅包含 attack_cat 列
    x_test: 测试集的特征数据，去除了标签列
    y_test: 测试集的标签数据，仅包含 attack_cat 列
    x_val: 验证集的特征数据，去除了标签列 attack_cat
    y_val: 验证集的标签数据，仅包含 attack_cat 列
    '''
    # 删除不需要的列
    data = data.drop(columns=['srcip','sport','dstip','dsport','Label'])
    
    # 划分数据集
    train, test = train_test_split(data, test_size=0.2, random_state=16)
    train, val = train_test_split(train, test_size=0.2, random_state=16)
    
    # 分离特征和标签
    x_train, y_train = train.drop(columns=['attack_cat']), train[['attack_cat']]
    x_test, y_test = test.drop(columns=['attack_cat']), test[['attack_cat']]
    x_val, y_val = val.drop(columns=['attack_cat']), val[['attack_cat']]
    
    return x_train, y_train, x_test, y_test, x_val, y_val

def preprocess_features(x_train, x_test, x_val):
    '''
    对训练、测试和验证集的数据进行预处理:
    特征划分：将特征划分为分类特征（cat_col）和数值特征（num_col）；
    标准化数值特征：使用 StandardScaler 对数值特征进行标准化
    编码分类特征：使用 OneHotEncoder 对分类特征进行独热编码
    数据重塑：将数据 reshape 成三维数组，适用于如 CNN 或 RNN 等模型输入
    
    x_train: 预处理前的训练集 
    x_test: 预处理前的测试集
    x_val: 预处理前的验证集
    '''
    cat_col = ['proto', 'service', 'state']     # 分类特征
    num_col = list(set(x_train.columns) - set(cat_col))    # 数值特征
    
    # 标准化数值特征
    scaler = StandardScaler()       # 对数值特征进行标准化处理
    scaler = scaler.fit(x_train[num_col])   # 在训练集上拟合模型并转换数据
    # 使用训练集的参数对测试集和验证集进行标准化，确保数据分布一致。
    x_train[num_col] = scaler.transform(x_train[num_col])
    x_test[num_col] = scaler.transform(x_test[num_col])
    x_val[num_col] = scaler.transform(x_val[num_col])
    
    # 编码分类特征
    ct = ColumnTransformer(
        transformers=[
            (
                'encoder', 
                OneHotEncoder(sparse=False),
                cat_col
            )
        ], 
        remainder='passthrough'
    )
    x_train = np.array(ct.fit_transform(x_train))
    x_test = np.array(ct.transform(x_test))
    x_val = np.array(ct.transform(x_val))
    
    # 重塑数据为3维，(样本数, 特征数, 通道数)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    
    return x_train, x_test, x_val

def preprocess_labels(y_train, y_test, y_val):
    '''
    对标签数据中的 'attack_cat' 列进行独热编码
    '''
    attacks = y_train['attack_cat'].unique()
    ct1 = ColumnTransformer(
        transformers=[
            (
                'encoder',
                OneHotEncoder(categories=[attacks],sparse=False),
                ['attack_cat']
            )
        ], 
        remainder='passthrough'
    )
    y_train = np.array(ct1.fit_transform(y_train))
    y_test = np.array(ct1.transform(y_test))
    y_val = np.array(ct1.transform(y_val))
    
    return y_train, y_test, y_val, attacks 