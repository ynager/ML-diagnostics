import sklearn as skl
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn import preprocessing
import pylab as plt
from scipy.ndimage.filters import gaussian_filter
import numpy as np


class Normalization(skl.base.BaseEstimator, skl.base.TransformerMixin):

    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        #self.normalizer = preprocessing.Normalizer()

    def fit(self, X, y):
        print("Scaler fitted")
        #self.scaler.fit(X)
        #self.normalizer.fit(X)
        return self

    def transform(self, X):
        X_shape = X.shape
        #X_new = self.scaler.fit_transform(X)
        #X_new = self.normalizer.transform(X)
        X_mean = X.mean(axis=1)
        X = X-X_mean.reshape(X_shape[0],1)
        test_max = X.max(axis=1)
        X_new = X/test_max.reshape(X_shape[0],1)
        print("Scaler applied")
        return X_new

class ReduceResolution(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, factor):
        self.factor = factor
    
    def fit(self, X, y=None):
        print("ReduceResolution fitted")
        return self
    
    def transform(self, X, y=None):
        X_new = X[:, 0::self.factor]
        print("ReduceResolution transformed")
        return X_new

class Histogram(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, n_bins=10, rmin=0, rmax=1024):
        self.n_bins = n_bins
        self.rmin = rmin
        self.rmax = rmax

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        H = np.zeros((X.shape[0], self.n_bins), dtype=np.int32)
        for samp in range(X_new.shape[0]):
            H[samp, :] = np.histogram(X[samp, :], bins=n_bins, range=(self.rmin,self.rmax))[0]
        return H


class Flatten(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """Flatten"""
    def __init__(self, dim=2):
        self.dim = dim
        self.min = 0
        self.max = 0
    
    def fit(self, X, y=None):
        self.min = X.min()
        self.max = X.max()
        return self
    
    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176) # Bad practice: hard-coded dimensions
        #X_new = gaussian_filter(X,(0,0,4,4)) #filter
        X_new = X[:,70:130]
        #X = X.mean(axis=self.dim)
        print('Flatten transform')

        X_new = X_new.reshape(X_new.shape[0], -1)
        return X_new
