from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from os.path import normpath
from sklearn.model_selection import StratifiedKFold


class GridSearchCV(GridSearchCV):
    """docstring for GridSearchCV"""

    def __init__(self, est_class, est_params, param_grid, cv=None,
                 n_jobs=40, pre_dispatch="3*n_jobs", error_score="raise",
                 save_path=None, **kwargs):

        self.est_class = est_class
        self.est_params = est_params
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.estimator = est_class(est_params)
        self.set_save_path(save_path)
        self.cv = cv
        if cv is not None and type(cv) is not int:
            self.cv_obj = cv["class"](**cv["params"])
        elif type(cv) is int:
            self.cv_obj = cv
        else:
            self.cv_obj = None
        super(GridSearchCV, self).__init__(self.estimator, param_grid,
                                           cv=self.cv_obj,
                                           n_jobs=self.n_jobs,
                                           refit=True,
                                           error_score=error_score,
                                           **kwargs)

    def fit(self, X, y=None, groups=None, **fit_params):
        super(GridSearchCV, self).fit(X, y, groups, **fit_params)

        if self.save_path is not None:
            data = {
                "best_params_": self.best_params_,
                "mean_test_score": self.cv_results_["mean_test_score"],
                "std_test_score": self.cv_results_["std_test_score"],
            }
            df = pd.DataFrame.from_dict(pd.io.json.json_normalize(data))
            df.to_csv(normpath(self.save_path+"GridSearchCV.csv"))

            if hasattr(self.best_estimator_, "save_path"):
                self.best_estimator_.set_save_path(self.save_path)
        print("*******************************")
        print("best params: {} " .format(self.best_params_))
        print("best score: {}" .format(self.best_score_))
        print("best index: {}" .format(self.best_index_))
        print("*******************************")

        return self

    def set_save_path(self, save_path):
        self.save_path = save_path
        if (hasattr(self, "best_estimator_") and
           hasattr(self.best_estimator_, "save_path")):
            self.best_estimator_.set_save_path(save_path)


class SKFold():
    def __init__(self, n_splits, shuffle):
        self.n_splits = n_splits
        self.shuffle = shuffle

        self.model = StratifiedKFold(n_splits=self.n_splits,
                                     shuffle=self.shuffle)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.model.get_n_splits(X, y)

    def split(self, X, y, groups=None):
        y = np.argmax(y, axis=1)
        return self.model.split(X, y)
