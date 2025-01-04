from sklearn.base import BaseEstimator, TransformerMixin

class AddConstantTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add a small constant to specified columns.
    """
    def __init__(self, constant=1, columns=None):
        self.constant = constant
        self.columns = columns  # Columns to apply the transformation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is not None:
            X[self.columns] += self.constant
        return X