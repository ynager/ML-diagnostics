from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement

#PCA
from sklearn.decomposition import PCA

class PrincipleComponentAnalysis(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000):
        self.n_components = n_components
        self.model = PCA(self.n_components)
    
    def fit(self, X, y=None):
        X = check_array(X)
        self.model.fit(X)
        return self
    
    def transform(self, X, y=None):
        check_is_fitted(self, ["model"])
        X = check_array(X)
        X_new = self.model.transform(X)
        return X_new

    def set_save_path(self, save_path):
        self.save_path = save_path

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
