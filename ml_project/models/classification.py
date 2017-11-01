import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy import stats


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
    
    def __init__(self,learning_rate=0.1, n_estimators=100, verbose=1, subsample=1, max_depth=3, sw = True):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.subsample = subsample
        self.max_depth = max_depth
        self.sw = sw
                 
                 
    def fit(self, X, y, sample_weight=None):
        
        self.model = GradientBoostingClassifier(learning_rate=self.learning_rate,
                                                n_estimators=self.n_estimators,
                                                subsample=self.subsample,
                                                verbose=self.verbose,
                                                max_depth=self.max_depth)
        
        if self.sw:
            Xnew = np.zeros((X.shape[0]*y.shape[1],X.shape[1]))
            ynew = np.zeros((X.shape[0]*y.shape[1]))
            w = np.zeros((X.shape[0]*y.shape[1]))
            
            for i in range(X.shape[0]):
                Xnew[y.shape[1]*i:y.shape[1]*i+y.shape[1],:] = X[i,:]
                ynew[y.shape[1]*i:y.shape[1]*i+y.shape[1]] = np.arange(y.shape[1])
                w[y.shape[1]*i:y.shape[1]*i+y.shape[1]] = y[i,:]
        
            print(X[0,0:50])
            print(Xnew[2,0:50])
            print(Xnew[3,0:50])
            print(y[0])
            print(ynew)
            print(w)
            self.model.fit(Xnew, ynew, w)

    
        else:
            y = np.argmax(y,axis=1)
            self.model.fit(X, y)
        
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

    def __init__(self, C=1, kernel='rbf', probability=True):
        self.probability = probability
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        self.model = SVC(C=self.C, kernel=self.kernel, probability=self.probability, verbose=True)
    
    
        if True:
            Xnew = np.zeros((X.shape[0]*y.shape[1],X.shape[1]))
            ynew = np.zeros((X.shape[0]*y.shape[1]))
            w = np.zeros((X.shape[0]*y.shape[1]))
            
            for i in range(X.shape[0]):
                Xnew[y.shape[1]*i:y.shape[1]*i+y.shape[1],:] = X[i,:]
                ynew[y.shape[1]*i:y.shape[1]*i+y.shape[1]] = np.arange(y.shape[1])
                w[y.shape[1]*i:y.shape[1]*i+y.shape[1]] = y[i,:]
    
            self.model.fit(Xnew, ynew, w)
        
        
        else:
            y = np.argmax(y,axis=1)
            self.model.fit(X, y)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        ypred = self.predict_proba(X)
        return np.mean(stats.spearmanr(ypred,y,axis=1).correlation)








