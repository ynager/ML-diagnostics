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
    
    def __init__(self,learning_rate=0.1, n_estimators=100, verbose=1, subsample=1, max_depth=3, p_threshold=1):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.subsample = subsample
        self.max_depth = max_depth
        self.p_threshold = p_threshold
                 
                 
    def fit(self, X, y, sample_weight=None):
        
        self.model = GradientBoostingClassifier(learning_rate=self.learning_rate,
                                                n_estimators=self.n_estimators,
                                                subsample=self.subsample,
                                                verbose=self.verbose,
                                                max_depth=self.max_depth)
        
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








