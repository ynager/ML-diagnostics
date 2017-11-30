import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
import math
from scipy.ndimage.filters import uniform_filter1d
# from sklearn.model_selection import KFold
# from ml_project.model_selection import SKFold


def crossvalscore(path_m, path_x, path_y, k, n_jobs=1):

    clf = joblib.load(path_m)
    X = np.load(path_x)
    y = np.loadtxt(path_y)

    # cv = KFold(n_splits=k, shuffle=True)
    cv = StratifiedKFold(y, k)
    print("Starting " + str(k) + "-fold cross-validation...")
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=n_jobs)
    print("Scores: {}".format(scores))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def make_blocks(X, t):
    xdim = range(0, X.shape[0], t)
    ydim = range(0, X.shape[1], t)
    zdim = range(0, X.shape[2], t)
    reshaped = np.zeros((X.shape[0]//t*X.shape[1]//t*X.shape[2]//t, t, t, t))
    block = 0
    for x in xdim:
        for y in ydim:
            for z in zdim:
                reshaped[block] = X[x:x+t, y:y+t, z:z+t]
                block += 1
    return reshaped


def anisodiff3(stack, niter=1, kappa=50, gamma=0.1, step=(1., 1., 1.),
               option=1, ploton=False):
    """
        3D Anisotropic diffusion.

        Usage:
        stackout = anisodiff(stack, niter, kappa, gamma, option)

        Arguments:
        stack  - input stack
        niter  - number of iterations
        kappa  - conduction coefficient 20-100 ?
        gamma  - max value of .25 for stability
        step   - tuple, the distance between adjacent pixels in (z,y,x)
        option - 1 Perona Malik diffusion equation No 1
        2 Perona Malik diffusion equation No 2
        ploton - if True, the middle z-plane will be plotted on every
        iteration

        Returns:
        stackout   - diffused stack.

        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence
        diffusionacross step edges.  A large value reduces the influence of
        intensity gradients on conduction.

        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)

        step is used to scale the gradients in case the spacing between
        adjacent pixels differs in the x,y and/or z axes

        Diffusion equation 1 favours high contrast edges over low contrast
        ones.
        Diffusion equation 2 favours wide regions over smaller ones.

        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.

        Original MATLAB code by Peter Kovesi
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>

        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>

        June 2000  original version.
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """

    if stack.ndim == 4:
        print("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        # from time import sleep

        showplane = stack.shape[0]//2

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(stack[showplane, ...].squeeze(), interpolation='nearest')
        ih = ax2.imshow(stackout[showplane, ...].squeeze(),
                        interpolation='nearest', animated=True)
        ax1.set_title("Original stack (Z = %i)" % showplane)
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in range(niter):

        # calculate the diffs
        deltaD[:-1, :, :] = np.diff(stackout, axis=0)
        deltaS[:, :-1, :] = np.diff(stackout, axis=1)
        deltaE[:, :, :-1] = np.diff(stackout, axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # update the image
        stackout += gamma*(UD+NS+EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(stackout[showplane, ...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
    # sleep(0.01)

    return stackout


def image_histogram_equalization(image, number_bins=256, max_value=255):
    # http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(),
                                         number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = max_value * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


class ecg_analysis:
    def __init__(self, dataset, hrw, fs):
        self.measures = {}
        self.dataset = dataset
        self.hrw = hrw
        self.fs = fs
    
        self.detect_peaks()
        self.calc_RR()

    def rolmean(self):
        mov_avg = uniform_filter1d(self.dataset, size=(int(self.hrw*self.fs)))
        avg_hr = (np.mean(self.dataset))
        mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
        return mov_avg

    def detect_peaks(self):
        window = []
        peaklist = []
        listpos = 0
        
        rollingmean_sig = self.rolmean()
        
        for datapoint in self.dataset:
            rollingmean = rollingmean_sig[listpos]
            if (datapoint <= rollingmean) and (len(window) <= 1): #Here is the update in (datapoint <= rollingmean)
                listpos += 1
            elif (datapoint > rollingmean):
                window.append(datapoint)
                listpos += 1
            else:
                maximum = max(window)
                beatposition = listpos - len(window) + (window.index(max(window)))
                peaklist.append(beatposition)
                window = []
                listpos += 1
        self.measures['peaklist'] = peaklist
        self.measures['ybeat'] = [self.dataset[x] for x in peaklist]

    def calc_RR(self):
        peaklist = self.measures['peaklist']
        RR_list = []
        cnt = 0
        while (cnt < (len(peaklist)-1)):
            RR_interval = (peaklist[cnt+1] - peaklist[cnt])
            ms_dist = ((RR_interval / self.fs) * 1000.0)
            RR_list.append(ms_dist)
            cnt += 1
    
        RR_diff = []
        RR_sqdiff = []
        cnt = 0
        
        while (cnt < (len(RR_list)-1)):
            RR_diff.append(abs(RR_list[cnt] - RR_list[cnt+1]))
            RR_sqdiff.append(math.pow(RR_list[cnt] - RR_list[cnt+1], 2))
            cnt += 1
        
        self.measures['RR_list'] = RR_list
        self.measures['RR_diff'] = RR_diff
        self.measures['RR_sqdiff'] = RR_sqdiff

    def calc_ts_measures(self):
        RR_list = self.measures['RR_list']
        RR_diff = self.measures['RR_diff']
        RR_sqdiff = self.measures['RR_sqdiff']
        self.measures['bpm'] = 60000 / np.mean(RR_list)
        self.measures['ibi'] = np.mean(RR_list)
        self.measures['sdnn'] = np.std(RR_list)
        self.measures['sdsd'] = np.std(RR_diff)
        self.measures['rmssd'] = np.sqrt(np.mean(RR_sqdiff))
        NN20 = [x for x in RR_diff if (x>20)]
        NN50 = [x for x in RR_diff if (x>50)]
        self.measures['nn20'] = NN20
        self.measures['nn50'] = NN50
        self.measures['pnn20'] = float(len(NN20)) / float(len(RR_diff))
        self.measures['pnn50'] = float(len(NN50)) / float(len(RR_diff))

    def calc_bpm(self):
        RR_list = self.measures['RR_list']
        self.measures['bpm'] = 60000 / np.mean(RR_list)

    def get_measures_array(self):

        m = self.measures
        list = [m['bpm'], m['ibi'], m['sdnn'], m['sdsd'], m['rmssd'], m['pnn20'], m['pnn50']]

        return list
