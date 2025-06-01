# CNN-LSTM 异常检测

这是一个使用CNN-LSTM混合模型进行网络流量异常检测的项目。该项目基于UNSW-NB15数据集，使用深度学习技术来识别网络流量中的异常行为。

## 项目结构

```
.
├── data_processing.py    # 数据加载和预处理
├── visualization.py      # 数据可视化
├── model.py             # 模型定义
├── train.py             # 模型训练和评估
├── requirements.txt     # 项目依赖
└── README.md           # 项目说明
```

## 环境要求

- Python 3.7+
- TensorFlow 2.4.1+
- 其他依赖见 requirements.txt

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/Pocon041/CNN-LSTM-Anomaly-Detection.git
cd CNN-LSTM-Anomaly-Detection
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：
   - 将UNSW-NB15数据集放在 `archive` 目录下
   - 确保数据文件命名为 `UNSW-NB15_1.csv` 到 `UNSW-NB15_4.csv`
   - 确保特征文件命名为 `NUSW-NB15_features.csv`
   - 数据集可以在https://research.unsw.edu.au/projects/unsw-nb15-dataset处下载

2. 运行训练：
```bash
python train.py
```

## 模型架构

该模型结合了CNN和LSTM的优点：
- 使用CNN提取空间特征
- 使用LSTM捕获时序依赖
- 包含多个卷积块和LSTM块
- 使用Dropout和BatchNormalization防止过拟合

## 评估指标

模型使用以下指标进行评估：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- 混淆矩阵
