import sklearn as skl
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from matplotlib import pyplot as plt


from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import BayesianRidge


class SVRegression(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""
    def __init__(self, C=20, epsilon=0.1, kernel='linear', save_path=None):
        super(SVRegression, self).__init__()
        self.save_path = save_path
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.model = None

    def fit(self, X, y):
        self.model = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        self.model.fit(X, y)
        print("SVR fitted")
        return self

    def predict(self, X):
        X = check_array(X)
        prediction = self.model.predict(X)
        print("SVR predicted")
        print(prediction)
        return prediction

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        if self.save_path is not None:
            plt.figure()
            plt.plot(scores, "o")
            plt.savefig(self.save_path + "SVRegressionScore.png")
            plt.close()

            df = pd.DataFrame({"score": scores})
            df.to_csv(self.save_path + "SVScore.csv")

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path


class RidgeRegression(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, alpha=100, save_path=None):
        super(RidgeRegression, self).__init__()
        self.save_path = save_path
        self.alpha = alpha
        self.model = None

    def fit(self, X, y):
        self.model = Ridge(alpha=self.alpha,
                             fit_intercept=True)

        self.model.fit(X, y)
        #print("Ridge fitted with alpha:")
        #print(self.model.alpha_)
        return self

    def predict(self, X):
        X = check_array(X)
        prediction = self.model.predict(X)
        print("Ridge predicted")
        np.save("prediction.npy",prediction)
        return prediction

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        if self.save_path is not None:
            plt.figure()
            plt.plot(scores, "o")
            plt.savefig(self.save_path + "RidgeScore.png")
            plt.close()

            df = pd.DataFrame({"score": scores})
            df.to_csv(self.save_path + "RidgeScore.csv")

        return -score

    def set_save_path(self, save_path):
        self.save_path = save_path

class HuberRegression(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, alpha=0.0001, max_iter=100, epsilon=1.35, save_path=None):
        super(HuberRegression, self).__init__()
        self.save_path = save_path
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = None
    
    def fit(self, X, y):
        self.model = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha, max_iter=self.max_iter)

        self.model.fit(X, y)

        return self

    def predict(self, X):
        X = check_array(X)
        prediction = self.model.predict(X)
        print("Huber predicted")
        return prediction
    
    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        return -score

    def set_save_path(self, save_path):
        self.save_path = save_path

class BayesianRidgeRegression(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, n_iter=300, save_path=None):
        super(BayesianRidgeRegression, self).__init__()
        self.save_path = save_path
        self.n_iter = n_iter
        self.model = None
    
    def fit(self, X, y):
        self.model = BayesianRidge(n_iter=self.n_iter, fit_intercept=True)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        X = check_array(X)
        prediction = self.model.predict(X)
        print("BayesianRidge predicted")
        return prediction
    
    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)
        return -score
    
    def set_save_path(self, save_path):
        self.save_path = save_path


class KernelEstimator(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""
    def __init__(self, save_path=None):
        super(KernelEstimator, self).__init__()
        self.save_path = save_path

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.y_mean = np.mean(y)
        y -= self.y_mean
        Xt = np.transpose(X)
        cov = np.dot(X, Xt)
        alpha, _, _, _ = np.linalg.lstsq(cov, y)
        self.coef_ = np.dot(Xt, alpha)

        if self.save_path is not None:
            plt.figure()
            plt.hist(self.coef_[np.where(self.coef_ != 0)], bins=50,
                     normed=True)
            plt.savefig(self.save_path + "KernelEstimatorCoef.png")
            plt.close()

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "y_mean"])
        X = check_array(X)

        prediction = np.dot(X, self.coef_) + self.y_mean

        if self.save_path is not None:
            plt.figure()
            plt.plot(prediction, "o")
            plt.savefig(self.save_path + "KernelEstimatorPrediction.png")
            plt.close()

        return prediction

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        if self.save_path is not None:
            plt.figure()
            plt.plot(scores, "o")
            plt.savefig(self.save_path + "KernelEstimatorScore.png")
            plt.close()

            df = pd.DataFrame({"score": scores})
            df.to_csv(self.save_path + "KernelEstimatorScore.csv")

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path


class GaussianProcessRegression(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, kernel=None, alpha=None, save_path=None):
        super(GaussianProcessRegression, self).__init__()
        self.save_path = save_path
        self.alpha = alpha
        self.kernel = kernel
        self.model = None

    def fit(self, X, y):
        self.model = GaussianProcessRegressor(alpha = self.alpha, kernel = self.kernel, normalize_y = True, n_restarts_optimizer=5)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        prediction = self.model.predict(X)
        print("Gaussian predicted")
        return prediction

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)
        return score

    def set_save_path(self, save_path):
        self.save_path = save_path
