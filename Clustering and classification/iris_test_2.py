from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
iris_datas = datasets.load_iris()  # 加载数据集
f_datas = iris_datas.data  # 得到特征数据
labels = iris_datas.target  # 得到标签数据
train_datas, test_datas, train_labels, test_labels = train_test_split(f_datas, labels, train_size=120)  # 划分训练集和测试集
knn = KNeighborsClassifier(n_neighbors=10)  # 创建KNN模型，指定邻居数
knn.fit(train_datas, train_labels)  # 拟合训练数据学习模型参数
predict_labels = knn.predict(test_datas)  # 用训练好的模型预测测试数据结果
print(predict_labels)  # 打印预测结果
print(test_labels)  # 打印真实结果
errors = np.nonzero(predict_labels - test_labels)  # 统计预测失败个数
print(errors)
print("预测失败的个数为：", len(errors[0]))