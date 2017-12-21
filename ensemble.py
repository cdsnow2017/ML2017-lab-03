import pickle
import numpy as np
from math import log

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        #基分类器类型
        self.weak_classifier = weak_classifier
        #基分类器的最大数目
        self.n_weakers_limit = n_weakers_limit
        #基分类器权重
        self.alpha_list = [] 
        #训练好的基分类器列表
        self.weak_classifier_list = []

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        X_num = X.shape[0]
        #初始化权值
        original_weights = 1.0 / X_num
        #权值列表
        self.W = np.linspace( original_weights, original_weights, X_num)
        for i in range(0,self.n_weakers_limit):
            base_classifier = self.weak_classifier(max_depth = 1)
            base_classifier.fit(X,y,sample_weight = self.W)
            y_predict = base_classifier.predict(X)
            #计算误差率em
            em = self.compute_em(y_predict,y)
            if em >= 0.5:
                break
            #基分类器的权重
            alpha = 0.5 * log((1.0 - em ) / max(em,1e-16))
            #计算规范化因子
            zm =  sum(self.W * np.exp(- alpha * y_predict * y))
            #更新样本的权重
            self.W = self.W * np.exp(- alpha * y_predict * y) / zm
            #把这次迭代的基分类器添加到结果中
            self.alpha_list.append(alpha)
            self.weak_classifier_list.append(base_classifier)
   
    def compute_em(self,y_predict,y):
        error_arr = (y_predict != y)
        error_arr = error_arr.astype('int32')
        em = sum(error_arr * self.W)
        return em
      
        
        


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        total_score = np.zeros(X.shape[0])
        for weak_classifier,alpha in zip(self.weak_classifier_list,self.alpha_list):
            y_predict = weak_classifier.predict(X)
            total_score += y_predict * alpha
        return total_score.reshape((len(total_score),1))

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        total_score = np.zeros(X.shape[0])
        for weak_classifier,alpha in zip(self.weak_classifier_list,self.alpha_list):
            y_predict = weak_classifier.predict(X)
            total_score += y_predict * alpha
        #计算最终的分类值
        result = np.ones(total_score.shape)  
        result[total_score < threshold]= -1
        return result.reshape((len(result),1))

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
