import numpy as np
from sklearn.model_selection import train_test_split

from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from data_process import load
from sklearn.metrics import classification_report

if __name__ == "__main__":
    #加载之前处理好的NPD
    face = load('dump/NPD_face')
    noface = load('dump/NPD_noface')
    #生成标签1，-1
    y_face = np.ones(face.shape[0])
    y_noface = np.ones(noface.shape[0]) *(-1)
    #将有脸和无脸数据融合，生成测试集和验证集
    X = np.concatenate((face,noface),axis=0)
    y = np.concatenate((y_face,y_noface),axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33)
    #构造一个40层的boost分类器
    ada = AdaBoostClassifier(DecisionTreeClassifier,40)
    #训练
    ada.fit(X_train,y_train)
    #预测
    y_test_predict= ada.predict(X_test,0)
    #输出报告
    labels =[1,-1] 
    target_names = ['face', 'nonface']
    repot_str = classification_report(y_test, y_test_predict,labels=labels,target_names=target_names)
    #写入报告
    output = open('report.txt', 'w')
    output.write(repot_str)
    output.close()
    pass

