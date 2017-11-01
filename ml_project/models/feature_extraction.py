from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Crop(BaseEstimator, TransformerMixin):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.reshape(-1, 176, 208, 176)
        X_new = X[:, self.xmin:self.xmax, self.ymin:self.ymax, self.zmin:self.zmax]
        X_new = X_new.reshape(X_new.shape[0], -1)

        return X_new

class Histogramize3d(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins, rmin, rmax):
        self.rmin = rmin
        self.rmax = rmax
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if (X.ndim == 5):
            H = np.zeros((X.shape[0], X.shape[1], self.n_bins), dtype=np.int32)
            for samp in range(X.shape[0]):
                for cub in range(X.shape[1]):
                    H[samp, cub, :] = np.histogram(X[samp,cub, :],
                                                   bins=self.n_bins,
                                                   range=(self.rmin,
                                                          self.rmax))[0]
        else:
            H = np.zeros((X.shape[0], self.n_bins), dtype=np.int32)
            for samp in range(X.shape[0]):
                H[samp, :] = np.histogram(X[samp, :],
                                          bins=self.n_bins,
                                          range=(self.rmin,
                                                 self.rmax))[0]


        Hflat = H.reshape(H.shape[0], -1)
        return Hflat
        
