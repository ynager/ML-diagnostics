import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy import stats
from sklearn.utils import resample


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""
    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))


from sklearn.ensemble import GradientBoostingClassifier
class GradientBoostingClassification(BaseEstimator, TransformerMixin):
    
    def __init__(self,learning_rate=0.1, n_estimators=100, verbose=1, subsample=1, max_depth=3, p_threshold=1):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.subsample = subsample
        self.max_depth = max_depth
        self.p_threshold = p_threshold
                 
                 
    def fit(self, X, y, sample_weight=None):
        print("X shape: {}" .format(X.shape))
        self.model = GradientBoostingClassifier(learning_rate=self.learning_rate,
                                                n_estimators=self.n_estimators,
                                                subsample=self.subsample,
                                                verbose=self.verbose,
                                                max_depth=self.max_depth)
        
        yn = np.argmax(y,axis=1)
        
        # create sample weights
        w = np.ones((X.shape[0]))
        
        if self.p_threshold < 1:
            for i in range(X.shape[0]):
                if(np.max(y[i]) < self.p_threshold):
                    w[i] = 0.001
                else:
                    w[i] = np.max(y[i])
        
        # upsample minority classed
        #X_0 = X[yn==0]
        #X_1 = X[yn==1]
        #X_2 = X[yn==2]
        #X_3 = X[yn==3]
        
        #w_0 = w[yn==0]
        #w_1 = w[yn==1]
        #w_2 = w[yn==2]
        #w_3 = w[yn==3]
        
        #X_1_up, w_1_up = resample(X_1, w_1, n_samples = X_0.shape[0], replace=True, random_state=123)
        #X_2_up, w_2_up = resample(X_2, w_2, n_samples = X_0.shape[0], replace=True, random_state=123)
        #X_3_up, w_3_up = resample(X_3, w_3, n_samples = X_0.shape[0], replace=True, random_state=123)
        
        #Xu = np.concatenate((X_0, X_1_up, X_2_up, X_3_up))
        #wu = np.concatenate((w_0, w_1_up, w_2_up, w_3_up))
        #yu = np.concatenate((0*np.ones(X_0.shape[0]), 1*np.ones(X_0.shape[0]), 2*np.ones(X_0.shape[0]), 3*np.ones(X_0.shape[0])))
        
        #self.model.fit(Xu, yu, wu)
        self.model.fit(X, yn, w)
        return self


    def predict(self, X):
        self.model.predict(X)

    def predict_proba(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred

    def score(self, X, y):
        ypred = self.predict_proba(X)
        return np.mean(stats.spearmanr(ypred,y,axis=1).correlation)

from sklearn.svm import SVC
class SupportVectorClassification(BaseEstimator, TransformerMixin):

    def __init__(self, C=1, kernel='rbf', probability=True, p_threshold=1):
        self.probability = probability
        self.kernel = kernel
        self.C = C
        self.p_threshold = p_threshold

    def fit(self, X, y):
        print("X shape: {}" .format(X.shape))
        self.model = SVC(C=self.C, kernel=self.kernel, probability=self.probability, class_weight='balanced')
    
    
        w = np.ones((X.shape[0]))
        yn = np.argmax(y,axis=1)
        
        if self.p_threshold < 1:
            for i in range(X.shape[0]):
                if(np.max(y[i]) < self.p_threshold):
                    w[i] = 0.001
                else:
                    w[i] = np.max(y[i])

        self.model.fit(X, yn, w)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        pred =  self.model.predict_proba(X)
        print(pred)
        return pred
    
    def score(self, X, y):
        ypred = self.predict_proba(X)
        return np.mean(stats.spearmanr(ypred,y,axis=1).correlation)


from sklearn.ensemble import RandomForestClassifier
class RandomForestClassification(BaseEstimator, TransformerMixin):

    def __init__(self, n_estimators=10, bootstrap=False, class_weight=None, p_threshold=1):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.p_threshold = p_threshold

    def fit(self, X, y):
        print("X shape: {}" .format(X.shape))
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, bootstrap=self.bootstrap, class_weight=self.class_weight, verbose=1)
        
        w = np.ones((X.shape[0]))
        yn = np.argmax(y,axis=1)
        
        if self.p_threshold < 1:
            for i in range(X.shape[0]):
                if(np.max(y[i]) < self.p_threshold):
                    w[i] = 0.001
                else:
                    w[i] = np.max(y[i])

        self.model.fit(X, yn, w)
        return self

    def set_params(**params):
        self.model.set_params(**params)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        pred =  self.model.predict_proba(X)
        print(pred)
        return pred
    
    def score(self, X, y):
        ypred = self.predict_proba(X)
        return np.mean(stats.spearmanr(ypred,y,axis=1).correlation)



