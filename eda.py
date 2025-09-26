import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._full_pipeline = None
        self.num_attribs = None
        self.cat_attribs = None

    def _preclean(self, X):
        # Reemplaza "?" sólo en columnas string/objeto
        X = X.copy()
        obj_cols = X.select_dtypes(include=["object"]).columns
        X[obj_cols] = X[obj_cols].replace("?", np.nan)
        if "income" in X.columns:
            X["income"] = X["income"].apply(lambda x: 1 if x == ">50K" else 0)
        return X

    def fit(self, X, y=None):
        Xc = self._preclean(X)
       
        self.num_attribs = Xc.select_dtypes(exclude=["object"]).columns.tolist()
        self.cat_attribs = Xc.select_dtypes(include=["object"]).columns.tolist()

        self._full_pipeline = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.num_attribs),
                ("cat", cat_pipeline, self.cat_attribs),
            ],
            verbose_feature_names_out=False
        )
        #Salida como DataFrame
        self._full_pipeline.set_output(transform="pandas")
        self._full_pipeline.fit(Xc)
        return self

    def transform(self, X, y=None):
        Xc = self._preclean(X)
        X_out = self._full_pipeline.transform(Xc)

        if self.num_attribs:
            X_out[self.num_attribs] = X_out[self.num_attribs].apply(pd.to_numeric, errors="coerce")
        if self.cat_attribs:
            X_out[self.cat_attribs] = X_out[self.cat_attribs].astype("object")

        return X_out

ACI = pd.read_csv('adult.csv')

data_preparer = DataFramePreparer().fit(ACI)
ACI_prepared = data_preparer.transform(ACI)

