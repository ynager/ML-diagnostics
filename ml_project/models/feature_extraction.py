from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from ml_project.models.utils import ecg_analysis
from scipy import signal

class AddECGMeasures(BaseEstimator, TransformerMixin):
    def __init__(self, hrw, fs):
        self.hrw = hrw
        self.fs = fs

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        
        #ECG Measures
        X_mes = np.zeros((X[1].shape[0], 7))
        for samp in range(X_mes.shape[0]):
            ecg = ecg_analysis(X[0][samp], self.hrw, self.fs)
            ecg.calc_ts_measures()
            X_mes[samp] = ecg.get_measures_array()
        
        
        Xaug = np.concatenate((X[1], X_mes), axis=1)
        
        
        return Xaug




class Crop(BaseEstimator, TransformerMixin):
    def __init__(self, xmin1, xmax1, ymin1, ymax1, zmin1, zmax1,
                 xmin2=None, xmax2=None, ymin2=None,
                 ymax2=None, zmin2=None, zmax2=None,
                 xmin3=None, xmax3=None, ymin3=None,
                 ymax3=None, zmin3=None, zmax3=None):

        self.v1 = (xmin1, xmax1, ymin1, ymax1, zmin1, zmax1)
        self.v2 = (xmin2, xmax2, ymin2, ymax2, zmin2, zmax2)
        self.v3 = (xmin3, xmax3, ymin3, ymax3, zmin3, zmax3)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.reshape(-1, 176, 208, 176)

        X_crop1 = X[:,
                    self.v1[0]:self.v1[1],
                    self.v1[2]:self.v1[3],
                    self.v1[4]:self.v1[5]]

        if self.v2[0] is not None:
            X_crop2 = X[:,
                        self.v2[0]:self.v2[1],
                        self.v2[2]:self.v2[3],
                        self.v2[4]:self.v2[5]]

            X_new = (X_crop1, X_crop2)
        else:
            X_new = (X_crop1, )

        if self.v3[0] is not None:
            X_crop3 = X[:,
                        self.v3[0]:self.v3[1],
                        self.v3[2]:self.v3[3],
                        self.v3[4]:self.v3[5]]

            X_new = (X_crop1, X_crop2, X_crop3)

        return X_new


class Histogramize3d(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins, rmin, rmax):
        self.rmin = rmin
        self.rmax = rmax
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if type(X) is np.ndarray:
            if (X.ndim == 5):
                H = np.zeros((X.shape[0], X.shape[1],
                              self.n_bins+3), dtype=np.int32)
                for samp in range(X.shape[0]):
                    for cub in range(X.shape[1]):
                        H[samp, cub, 0:self.n_bins] = \
                            np.histogram(X[samp, cub, :],
                                         bins=self.n_bins,
                                         range=(self.rmin,
                                         self.rmax))[0]

                        H[samp, cub, self.n_bins] = np.mean(X[samp, cub])
                        H[samp, cub, self.n_bins+1] = np.median(X[samp, cub])
                        H[samp, cub, self.n_bins+2] = np.var(X[samp, cub])
            else:
                H = np.zeros((X.shape[0], self.n_bins), dtype=np.int32)
                for samp in range(X.shape[0]):
                    H[samp, :] = np.histogram(X[samp, :],
                                              bins=self.n_bins,
                                              range=(self.rmin,
                                                     self.rmax))[0]
        else:
            H = np.zeros((X[0].shape[0], len(X),
                          self.n_bins+3), dtype=np.int32)

            for reg in range(len(X)):
                for samp in range(X[reg].shape[0]):
                    H[samp, reg, 0:self.n_bins] = \
                        np.histogram(X[reg][samp, :],
                                     bins=self.n_bins,
                                     range=(self.rmin,
                                     self.rmax))[0]

                    H[samp, reg, self.n_bins] = np.mean(X[reg][samp])
                    H[samp, reg, self.n_bins+1] = np.median(X[reg][samp])
                    H[samp, reg, self.n_bins+2] = np.var(X[reg][samp])

        Hflat = H.reshape(H.shape[0], -1)

        return Hflat


class saveX(BaseEstimator, TransformerMixin):
    def __init__(self, filename):
        self.filename = filename

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('X saved')
        np.save(self.filename + '.npy', X)
        return X
