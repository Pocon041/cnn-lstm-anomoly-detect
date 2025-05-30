import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from data_processing import load_data, clean_data, prepare_data, preprocess_features, preprocess_labels
from visualization import plot_correlation_matrix, plot_class_distribution, plot_confusion_matrix
from model import create_model

def train_and_evaluate():
    # 加载和预处理数据
    data = load_data()
    data = clean_data(data)
    x_train, y_train, x_test, y_test, x_val, y_val = prepare_data(data)
    
    # 特征预处理
    x_train, x_test, x_val = preprocess_features(x_train, x_test, x_val)
    y_train, y_test, y_val, attacks = preprocess_labels(y_train, y_test, y_val)
    
    # 创建和训练模型
    n_features = x_train.shape[1]
    model = create_model(n_features)
    
    # 训练模型
    history = model.fit(x_train, y_train, 
                       epochs=1, 
                       batch_size=512,
                       validation_data=(x_val, y_val))
    
    # 评估模型
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    
    # 交叉验证
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    y_train_labels = np.argmax(y_train, axis=1)
    
    scores = []
    for train_index, val_index in kfold.split(x_train, y_train_labels):
        X_train_inner, X_val_inner = x_train[train_index], x_train[val_index]
        y_train_inner, y_val_inner = y_train[train_index], y_train[val_index]
        
        model = create_model(n_features)
        model.fit(X_train_inner, y_train_inner, 
                 epochs=1, 
                 batch_size=256,
                 validation_data=(X_val_inner, y_val_inner))
        
        test_loss, test_acc, precision, recall = model.evaluate(x_val, y_val)
        scores.append([test_loss, test_acc, precision, recall])
    
    print("\nAverage K-Fold Cross-Validation Results (on Validation Set):")
    print("Loss:", np.mean([score[0] for score in scores]))
    print("Accuracy:", np.mean([score[1] for score in scores]))
    print("Precision:", np.mean([score[2] for score in scores]))
    print("Recall:", np.mean([score[3] for score in scores]))
    
    # 生成混淆矩阵
    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred_labels)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, attacks)

if __name__ == "__main__":
    train_and_evaluate() 