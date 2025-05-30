import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_matrix(data, num_col):
    correlation_matrix = data[num_col].corr()
    
    plt.figure(figsize=(12, 12))
    sns.heatmap(correlation_matrix, cmap='coolwarm', square=True)
    
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title('Correlation Heatmap of Features')
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.show()

def plot_class_distribution(labels):
    sns.countplot(x=labels)
    
    plt.xlabel('Class Label')
    plt.ylabel('Number of Data Points')
    plt.title('Class Distribution')
    
    plt.xticks(rotation=90)
    plt.show()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(label='Count')
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=10)
            
    plt.xticks(range(len(class_names)), class_names, rotation=90, ha='right')
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.show() 