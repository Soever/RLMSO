from ucimlrepo import fetch_ucirepo 
import pandas as pd
# fetch dataset 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score, classification_report
from RLMSO2 import RLMSO
# from RLMSO import RLMSO
# df = pd.read_csv('./data/wine_quality/winequality-red.csv', sep=';')
# X = df.iloc[:, :-1].to_numpy()
# # 提取目标值 (最后一列)
# y = df.iloc[:, -1].to_numpy()

# print("\nClassification Report:\n", classification_report(y_test, y_pred))
if __name__ == '__main__':
    rlmso =RLMSO()
    df = pd.read_csv('./data/wine_quality/winequality-red.csv', sep=';')
    X = df.iloc[:, :-1].to_numpy()
    # 提取目标值 (最后一列)
    y = df.iloc[:, -1].to_numpy()
    def f(x):
        X_binary = rlmso.binary(x)
        print(X_binary)
        if X_binary.sum() == 0:
            return 1
        
        X_selected = X[:, X_binary == 1]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        # 初始化支持向量机模型 (使用线性核)
        svm_model = SVC(kernel='linear')
        # 训练模型
        svm_model.fit(X_train, y_train)
        # 对测试集进行预测
        y_pred = svm_model.predict(X_test)
        # 评估模型
        accuracy = accuracy_score(y_test, y_pred)
        return 1-accuracy
    # def fobj(x):
    #     return x[0]**2+x[1]**2
    # f =  fobj
    Xfood,fval,gbest_t = rlmso.optimize(100,1000,-10,10,11,f)

