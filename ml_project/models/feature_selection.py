from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


class KPCA(BaseEstimator, TransformerMixin):
    def __init__(self, kernel='linear', is_on=1):
        self.is_on = is_on
        self.kernel = kernel
        self.model = KernelPCA(kernel=self.kernel)

    def fit(self, X, y=None):
        if(self.is_on == 1):
            X = check_array(X)
            self.model.fit(X)
            print("PCA fitted")
        return self

    def transform(self, X, y=None):
        if(self.is_on == 1):
            X_new = self.model.transform(X)
            print("PCA transformed")
            return X_new
        else:
            return X


class VarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        self.model = VarianceThreshold()
        self.model.fit(X)
        print("VarianceThreshold fitted")
        return self

    def transform(self, X, y=None):
        X_new = self.model.transform(X)
        print("VarianceThreshold transformed")
        return X_new


class NonZeroSelection(BaseEstimator, TransformerMixin):
    """Select non-zero voxels"""
    def fit(self, X, y=None):
        X = check_array(X)
        self.nonzero = X.sum(axis=0) > 0

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["nonzero"])
        X = check_array(X)
        return X[:, self.nonzero]


class KBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=50):
        self.k = k
        self.model = None

    def fit(self, X, y):
        self.model = SelectKBest(f_regression, k=self.k)
        yn = np.argmax(y, axis=1)
        self.model.fit(X, yn)
        return self

    def transform(self, X, y=None):
        X_new = self.model.transform(X)
        return X_new


class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new
