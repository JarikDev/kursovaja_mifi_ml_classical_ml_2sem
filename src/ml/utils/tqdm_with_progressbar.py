from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class TQDMWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=None, desc="Step"):
        self.transformer = transformer
        self.desc = desc

    def fit(self, X, y=None):
        with tqdm(total=1, desc=f"{self.desc} - fit") as pbar:
            self.transformer.fit(X, y)
            pbar.update()
        return self

    def transform(self, X):
        with tqdm(total=1, desc=f"{self.desc} - transform") as pbar:
            X_t = self.transformer.transform(X)
            pbar.update()
        return X_t

    def get_params(self, deep=True):
        out = dict(transformer=self.transformer, desc=self.desc)
        if deep and hasattr(self.transformer, 'get_params'):
            for key, value in self.transformer.get_params(deep=True).items():
                out[f"transformer__{key}"] = value
        return out

    def set_params(self, **params):
        transformer_params = {}
        for key, value in params.items():
            if key.startswith("transformer__"):
                transformer_key = key.split("__", 1)[1]
                transformer_params[transformer_key] = value
            else:
                setattr(self, key, value)
        if transformer_params:
            self.transformer.set_params(**transformer_params)
        return self
