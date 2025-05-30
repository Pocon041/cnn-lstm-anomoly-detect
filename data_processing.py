import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data():
    dfs = []
    for i in range(1,5):
        path = 'archive/UNSW-NB15_{}.csv'
        dfs.append(pd.read_csv(path.format(i), header=None))
    combined_data = pd.concat(dfs).reset_index(drop=True)
    
    dataset_columns = pd.read_csv('archive/NUSW-NB15_features.csv', encoding='ISO-8859-1')
    combined_data.columns = dataset_columns['Name']
    return combined_data

def clean_data(data):
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
    cat_col = ['proto', 'service', 'state']
    num_col = list(set(x_train.columns) - set(cat_col))
    
    # 标准化数值特征
    scaler = StandardScaler()
    scaler = scaler.fit(x_train[num_col])
    x_train[num_col] = scaler.transform(x_train[num_col])
    x_test[num_col] = scaler.transform(x_test[num_col])
    x_val[num_col] = scaler.transform(x_val[num_col])
    
    # 编码分类特征
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), cat_col)], remainder='passthrough')
    x_train = np.array(ct.fit_transform(x_train))
    x_test = np.array(ct.transform(x_test))
    x_val = np.array(ct.transform(x_val))
    
    # 重塑数据为3维
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    
    return x_train, x_test, x_val

def preprocess_labels(y_train, y_test, y_val):
    attacks = y_train['attack_cat'].unique()
    ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(categories=[attacks],sparse=False), ['attack_cat'])], remainder='passthrough')
    y_train = np.array(ct1.fit_transform(y_train))
    y_test = np.array(ct1.transform(y_test))
    y_val = np.array(ct1.transform(y_val))
    
    return y_train, y_test, y_val, attacks 