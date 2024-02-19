from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

if __name__ == '__main__':
    iris = load_iris()
    data = iris.get("data")
    target = iris.get("target")
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(x_train, y_train)
    train_score = KNN.score(x_train, y_train)   #训练集准确率
    test_score = KNN.score(x_test, y_test)   #测试集准确率
    print("模型的准确率：", test_score)
    X1 = np.array([[1.5, 3, 5.8, 2.2], [6.2, 2.9, 4.3, 1.3]])     #待预测数据：X1=[[1.5 , 3 , 5.8 , 2.2], [6.2 , 2.9 , 4.3 , 1.3]]
    prediction = KNN.predict(X1)        # 进行预测
    k = iris.get("target_names")[prediction]      # 种类名称
    print("第一朵花的种类为：", k[0])
    print("第二朵花的种类为：", k[1])