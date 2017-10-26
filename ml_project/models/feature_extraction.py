class Crop(skl.base.BaseEstimator, skl.base.TransformerMixin):
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
        X_new = X[:, xmin:xmax, ymin:ymax, zmin:zmax]
        X_new = X_new.reshape(X_new.shape[0], -1)
        return X_new
