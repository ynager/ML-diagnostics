from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


class PrincipleComponentAnalysis(BaseEstimator, TransformerMixin):
    """PCA dimensionality reduction"""
    def __init__(self, n_components=1000):
        self.n_components = n_components
        self.model = None

    def fit(self, X, y=None):
        self.model = PCA(whiten=False, n_components=self.n_components)
        X = check_array(X)
        self.model.fit(X)
        print("PCA fitted")
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["model"])
        X = check_array(X)
        X_new = self.model.transform(X)
        print("PCA transformed")
        return X_new


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

class SelectK(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None
    
    def fit(self, X, y=None):
        self.model = SelectKBest(f_regression, k=10)
        self.model.fit(X, y)
        print("SelectKBest fitted")
        return self
    
    def transform(self, X, y=None):
        X_new = self.model.transform(X)
        print("SelectKBest transformed")
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
