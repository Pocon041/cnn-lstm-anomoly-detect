import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dense, Dropout

'''
结构为：
Input  -> Conv1D -> MaxPooling -> BatchNorm -> LSTM -> ...
       -> Conv1D -> MaxPooling -> BatchNorm -> LSTM -> ...
       -> Conv1D -> MaxPooling -> BatchNorm -> LSTM
                     ↓
                Dense(64) -> Dropout -> Dense(10)
'''


def create_model(n_features):
    # 创建简单线性堆叠网络
    model = Sequential()

    # 第一个卷积块
    model.add(
        Conv1D(             # 一维卷积层
            filters=16,     # 卷积核个数
            kernel_size=1,  # 卷积核大小。由于使用了很小的卷积核，更多依赖于后续的LSTM层来捕捉序列中的时间动态特征。
            activation='relu',  #激活函数
            input_shape=(n_features,1)  # 输入数据的形状，n_features是特征数量
        )
    )
    model.add(
        MaxPooling1D(
            pool_size=2 # 池化窗口大小2，每两个连续的特征值会被合并成最大的那个值。
        )
    )    
    model.add(BatchNormalization())     # 批归一化层对池化后的输出进行标准化处理

    # 第一个LSTM层。接取第一个卷积块提取的特征序列作为输入
    model.add(
        LSTM(
            units=16,       # 神经元数量
            return_sequences=True   # 这样后续还能继续堆叠另一个LSTM层
        )
    )

    # 第二个卷积层
    model.add(
        Conv1D(
            filters=32, 
            kernel_size=3, 
            activation='relu'
        )
    )
    model.add(
        MaxPooling1D(
            pool_size=2
        )
    )
    model.add(BatchNormalization())

    # 第二个LSTM层
    model.add(LSTM(units=32, return_sequences=True))

    # 第三个卷积块
    model.add(
        Conv1D(
            filters=64, 
            kernel_size=5, 
            activation='relu'
        )
    )
    model.add(
        MaxPooling1D(
            pool_size=2
        )
    )
    model.add(BatchNormalization())

    # 第三个LSTM层
    model.add(LSTM(units=64))

    # 全连接层
    
    # 隐含层
    model.add(
        Dense(
            64,     # 神经元数量
            activation='relu'   
        )
    )
    model.add(
        Dropout(0.2)    #随机“关闭” 20% 的神经元
    )
    # 输出层
    model.add(
        Dense(
            10,     # 共十种异常类型
            activation='softmax'    # 归一化，适用于多分类问题
        )
    )
    
    model.compile(
        optimizer='adam',     # adam优化器
        loss='categorical_crossentropy',   # 使用分类交叉熵作为损失函数    https://blog.csdn.net/qq_35599937/article/details/105608354
        metrics=[
            'accuracy',                       # 评估标准包括准确率，精确率，召回率
            tf.keras.metrics.Precision(), 
            tf.keras.metrics.Recall()
        ]
    )
    
    return model 