from sklearn.base import BaseEstimator, clone
from mne.decoding import SlidingEstimator


class WindowParamEstimator(BaseEstimator):
    def __init__(
        self,  # ONE instance is given to SlidingEstimator
        base_cls,
        params_per_window: list[dict],
    ):
        self.base_cls = base_cls
        self.params_per_window = params_per_window
        self._counter = -1  # will be incremented on each .fit

    def fit(self, X, y):
        # SlidingEstimator clones **then** calls fit => clones inherit _counter
        self._counter += 1  # window index
        par = self.params_per_window[self._counter]
        k = par.pop("k")  # feature-count

        # Build actual pipeline for *this* window
        from sklearn.pipeline import make_pipeline
        from sklearn.feature_selection import SelectKBest, f_classif

        pipe = make_pipeline(
            SelectKBest(f_classif, k=k),
            self.base_cls().set_params(**par),
        )
        if hasattr(self.fitted_, "classes_"):
            self.classes_ = self.fitted_.classes_
        else:  # should never happen for classifiers
            raise AttributeError(
                "Inner model does not expose `classes_` – "
                "required by cross_val_predict / SlidingEstimator."
            )
        return self

    # These just delegate
    def predict_proba(self, X):
        return self.fitted_.predict_proba(X)

    def predict(self, X):
        return self.fitted_.predict(X)
