import sklearn as skl
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn import preprocessing


class Normalization(skl.base.BaseEstimator, skl.base.TransformerMixin):

    def __init__(self):
        self.model = preprocessing.StandardScaler()
    
    def fit(self, X, y):
        self.model.fit(X)
        print("Preprocessing fitted:")
        return self
    
    
    def transform(self, X):
        X_new = self.model.transform(X)
        
        return X_new
