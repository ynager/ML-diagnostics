import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy import stats
from sklearn.metrics import f1_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


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

class XGBClassification(BaseEstimator, TransformerMixin):
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 n_jobs=1, subsample=1, silent=False, test_size=0, esr=5,
                 reg_alpha=0, reg_lambda=1, gamma=0, min_child_weight=1):

        self.test_size = test_size
        self.esr = esr
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.subsample = subsample
        self.silent = silent
        self.test_size = test_size
        self.esr = esr
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
    

    def fit(self, X, y, sample_weight=None):
        
        self.model = XGBClassifier(max_depth=self.max_depth,
                                   learning_rate=self.learning_rate,
                                   objective='multi:softmax',
                                   n_estimators=self.n_estimators,
                                   subsample=self.subsample,
                                   silent=self.silent,
                                   reg_alpha=self.reg_alpha,
                                   reg_lambda=self.reg_lambda,
                                   gamma=self.gamma,
                                   min_child_weight=self.min_child_weight)
        
        if(self.test_size > 0):
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=self.test_size,
                                                                stratify=y)
            eval_set = [(X_test, y_test)]
            self.model.fit(X_train, y_train,
                           eval_metric="merror",
                           eval_set=eval_set,
                           verbose=True,
                           early_stopping_rounds=self.esr)
                           
            print("Eval results: ")
            print(self.model.evals_result())
        else:
            self.model.fit(X, y, )
        return self

    def predict(self, X):
        pred = self.model.predict(X)
        return pred

    def score(self, X, y):
        ypred = self.predict(X)
        return f1_score(y, ypred, average='micro')




class GradientBoostingClassification(BaseEstimator, TransformerMixin):

    def __init__(self, learning_rate=0.1, n_estimators=100,
                 verbose=1, subsample=1, max_depth=3, loss='deviance',
                 min_samples_split=2, min_samples_leaf=1):

        self.lr = learning_rate
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.subsample = subsample
        self.max_depth = max_depth
        self.loss = loss
        self.m_s_s = min_samples_split
        self.m_s_l = min_samples_leaf
    
        self.model = GradientBoostingClassifier(learning_rate=self.lr,
                                                n_estimators=self.n_estimators,
                                                subsample=self.subsample,
                                                verbose=self.verbose,
                                                max_depth=self.max_depth,
                                                loss=self.loss,
                                                min_samples_split=self.m_s_s,
                                                min_samples_leaf=self.m_s_l)

    def fit(self, X, y, sample_weight=None):
        print("X shape before classification: {}" .format(X.shape))



        self.model.fit(X, y)

        return self

    def predict(self, X):
        ypred = self.model.predict(X)
        return ypred.astype(int)

    def predict_proba(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred

    def score(self, X, y):
        ypred = self.predict(X)
        print(ypred)
        return f1_score(y, ypred, average='micro')


class SupportVectorClassification(BaseEstimator, TransformerMixin):

    def __init__(self, C=1, kernel='linear', probability=False, class_weight=None):
        self.probability = probability
        self.kernel = kernel
        self.C = C
        self.class_weight = class_weight

    def fit(self, X, y):
        print("X shape: {}" .format(X.shape))
        self.model = SVC(C=self.C, kernel=self.kernel,
                         probability=self.probability,
                         class_weight=self.class_weight)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        pred = self.model.predict(X)
        print(pred)
        return pred

    def predict_proba(self, X):
        pred = self.model.predict_proba(X)
        print("Prediction: " + str(pred))
        return pred

    def score(self, X, y):
        ypred = self.predict(X)
        print(ypred)
        return f1_score(y, ypred, average='micro')


class RandomForestClassification(BaseEstimator, TransformerMixin):

    def __init__(self, n_estimators=10, bootstrap=False, class_weight=None):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.class_weight = class_weight

    def fit(self, X, y):
        print("X shape: {}" .format(X.shape))
        self.model = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                            bootstrap=self.bootstrap,
                                            class_weight=self.class_weight,
                                            verbose=1)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        pred = self.model.predict_proba(X)
        print("Prediction: " + str(pred))
        return pred

    def score(self, X, y):
        ypred = self.predict(X)
        return f1_score(y, ypred, average='micro')


class MLPClassification(BaseEstimator, TransformerMixin):
    def __init__(self,
                 hidden_layer_sizes1=100,
                 hidden_layer_sizes2=1,
                 activation='relu',
                 solver='adam',
                 lr_init = 0.001,
                 alpha='0.0001',
                 learning_rate='constant',
                 max_iter=200,
                 verbose=False,
                 early_stopping=False,
                 validation_fraction=0.1):

        self.hidden_layer_sizes = (200, 200, 200, 200, 200)
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = lr_init
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                  activation=self.activation,
                                  solver=self.solver,
                                  alpha=self.alpha,
                                  learning_rate=self.learning_rate,
                                  max_iter=self.max_iter,
                                  verbose=self.verbose,
                                  early_stopping=self.early_stopping,
                                  validation_fraction=self.validation_fraction)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        pred = self.model.predict(X)
        return pred

    def score(self, X, y):
        ypred = self.predict(X)
        return f1_score(y, ypred, average='micro')
