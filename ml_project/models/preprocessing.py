import sklearn as skl
from sklearn.utils.validation import check_array
import scipy.ndimage as nd
import numpy as np
from ml_project.models.utils import make_blocks, anisodiff3
from scipy import signal
from multiprocessing import Pool



class Trim(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        X = X[:,self.min:self.max]
        return X

class WelchMethod(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, window='hamming', nperseg=256, cutoff_freq_h=15, cutoff_freq_l=0.5, fs=300, peak_load=False, nr_bins=50):
        self.window = window
        self.nperseg = nperseg
        self.cutoff_freq_h = cutoff_freq_h
        self.cutoff_freq_l = cutoff_freq_l
        self.fs = fs
        self.peak_load = peak_load
    
        #Peakfinder
        self.peakRange = [5,30,1]
        self.valleyRange = [20,80,1]
        self.nr_bins = nr_bins

    def fit(self, X, y):
        return self
    
    def doOneSample(self, samp):
        print(samp)
        max = 20000
        min = 0
        #min = np.where(X[samp,0:500] > 0)
        #min = min[0][0]
            
        for i in range(min+1000, self.X.shape[1]):
            if np.count_nonzero(self.X[samp, i:i+300:10]) == 0:
                max = i
                break
            
        #filter
        Xfiltered = signal.lfilter(self.b, self.a, self.X[samp, min:max])
        Xcrop = self.X[samp, min:max]
            
        #welch
        f, W = signal.welch(Xfiltered,fs=self.fs,window=self.window, nperseg=self.nperseg)
            
        #peakfinder
        X_peak = np.zeros((2, self.nr_bins))
            
        peaks = signal.find_peaks_cwt(Xcrop[0:max],range(self.peakRange[0],self.peakRange[1],self.peakRange[2]))
            
        if(len(peaks) != 0):
            m = np.mean(Xcrop[peaks])
            temp = np.where(Xcrop[peaks] > 0.9*m)
            peaks = peaks[temp]
            X_peak[0] = np.histogram(Xcrop[peaks], bins=self.nr_bins)[0]
            X_peak[1] = np.histogram(peaks[1:]-peaks[:-1], bins=self.nr_bins)[0]
        
        
        X_peak = X_peak.reshape(-1)
        X_tot = np.concatenate((W, X_peak))
        return X_tot
    
    

    def transform(self, X, y=None):
        self.X = X
        
        # build butterworth filter
        fs = self.fs
        nyq = 0.5 * fs
        n_cutoff_h = self.cutoff_freq_h / nyq
        n_cutoff_l = self.cutoff_freq_l / nyq
        self.b, self.a = signal.butter(5, [n_cutoff_l, n_cutoff_h], btype='band', analog=False)
            
        
        #Filter and Welch and peakfinder
        X_new = np.zeros((X.shape[0], self.nperseg//2+1 + self.nr_bins*2))
    
        pool = Pool(8)
        X_new_list = pool.map(self.doOneSample, range(X.shape[0]))
        X_new = np.asarray(X_new_list)

        return (X,X_new)



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


class GaussianFilter(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return nd.gaussian_filter(X, self.sigma)


class AnisotropicDiffusion(skl.base.BaseEstimator, skl.base.TransformerMixin):

    def __init__(self, kappa=50, niter=1):
            self.kappa = kappa
            self.niter = niter

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("running Anisotropi Diffusion...")
        if (self.niter == 0):
            return X

        for sec in range(len(X)):
            print("running aniso for sec " + str(sec))
            for f in range(X[0].shape[0]):
                X[sec][f] = anisodiff3(X[sec][f],
                                       option=1,
                                       kappa=self.kappa,
                                       niter=self.niter)
        return X


class TransformToCubes(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, d):
        self.d = d

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):

        num_cubes = np.zeros(len(X), dtype=np.int)
        for seg in range(len(X)):
            num_cubes[seg] = X[seg].shape[1]//self.d * \
                             X[seg].shape[2]//self.d * X[seg].shape[3]//self.d

            print("# cubes: {} " .format(num_cubes[seg]))

        Xnew = np.zeros((X[0].shape[0],
                         num_cubes[0],
                         self.d,
                         self.d,
                         self.d))

        for samp in range(X[0].shape[0]):
            Xnew[samp] = make_blocks(X[0][samp], self.d)

        for seg in range(1, len(X)):
            Xseg = np.zeros((X[seg].shape[0],
                             num_cubes[seg],
                             self.d,
                             self.d,
                             self.d))

            for samp in range(X[seg].shape[0]):
                Xseg[samp] = make_blocks(X[seg][samp], self.d)

            Xnew = np.concatenate((Xnew, Xseg), axis=1)

        print("X shape after cubing: {}" .format(Xnew.shape))
        return Xnew


class CropCubeHist(skl.base.BaseEstimator, skl.base.TransformerMixin):
    def __init__(self, rmin=0, rmax=4000, nbins=30, d=10,
                 anis_kappa=35, anis_niter=2):

        self.rmin = rmin
        self.rmax = rmax
        self.nbins = nbins
        self.d = d
        self.anis_kappa = anis_kappa
        self.anis_niter = anis_niter

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)

        # crop data
        # X = X[:, 48:128, 30:150, 30:150]
        X = X[:, 58:118, 70:150, 30:110]

        for f in range(X.shape[0]):
            X[f] = anisodiff3(X[f],
                              option=1,
                              kappa=self.anis_kappa,
                              niter=self.anis_niter)

        # bins
        n_bins = self.nbins

        # divide into cubes of size d
        d = self.d
        p, m, n = X[0].shape

        # just to get size
        Temp = X[0].reshape(-1, m//d, d, n//d, d).transpose(1, 3, 0, 2, 4) \
            .reshape(-1, d, d, d)

        H = np.zeros((X.shape[0], Temp.shape[0], n_bins), dtype=np.int32)

        for samp in range(X.shape[0]):
            Cubes = make_blocks(X[samp], d)
            for cub in range(Cubes.shape[0]):
                H[samp, cub, :] = np.histogram(Cubes[cub, :],
                                               bins=n_bins,
                                               range=(self.rmin,
                                                      self.rmax))[0]

        Hflat = H.reshape(H.shape[0], -1)
        print("dim Hflat: {}".format(Hflat.shape))
        return Hflat
