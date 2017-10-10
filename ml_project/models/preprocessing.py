import sklearn as skl
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn import preprocessing


class Normalization(skl.base.BaseEstimator, skl.base.TransformerMixin):

    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.normalizer = preprocessing.Normalizer()

    def fit(self, X, y):
        self.scaler.fit(X)
        #self.normalizer.fit(X)
        print("Scaler fitted")
        return self

    def transform(self, X):
        X_new = self.scaler.fit_transform(X)
        #X_new = self.normalizer.transform(X)
        print("Scaler applied")
        return X_new
