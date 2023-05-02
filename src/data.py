from pandas.core.frame import DataFrame
from typing import Dict, Union

import pandas as pd
import numpy as np 
import torch

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import deep_tabular_augmentation as dta

class Data: 
    """Data object used as reference for tabular data generation.
    
    Args:
        cfg (Dict): Configuration dictionary.
        
    Details: split data into train and test sets, preprocess data, and create data loaders."""

    def __init__(self, cfg: Dict):      
        self.target = cfg["target"]
        self.classification = cfg["classification"]
        self.test_prop = cfg["test_prop"]
        self.batch_size = cfg["batch_size"]        

        if self.classification: 
            self.target_class = cfg["target_class"]
        else: 
            raise NotImplementedError("Regression is not yet implemented.")

        self.feature_types = {
            "numeric": [col for col in cfg["numeric"] if col != self.target],
            "category": [col for col in cfg["category"] if col != self.target]
        }

        self.X_tr, self.X_te, self.y_tr, self.y_te = None, None, None, None
        self.train_loader, self.test_loader = None, None

    def __repr__(self) -> str:
        if self.classification: 
            return f"Data(target={self.target}, classification={self.classification}, reference={self.target_class}, test_prop={self.test_prop}, batch_size={self.batch_size})"
        else:
            return f"Data(target={self.target}, classification={self.classification}, test_prop={self.test_prop}, batch_size={self.batch_size})"

    def _make_preprocessor(self): 

        self._columns_to_scale = self.feature_types["numeric"]
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        self._columns_to_encode = self.feature_types["category"]
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")) 
            ]
        )

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self._columns_to_scale),
                ("cat", categorical_transformer, self._columns_to_encode)
            ]
        )

    def preprocess(self, df): 
        """Preprocess original data and split into train and test sets."""

        n_unique_values = df[self.target].nunique()
        if n_unique_values > 2:
            raise NotImplementedError("Multiclass classification is not yet implemented.")

        self.X = df.drop(columns=[self.target])

        if self.classification: 
            self.y = df[self.target].apply(lambda x: 1 if x==self.target_class else 0)

        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size=self.test_prop, random_state=42)

        self._make_preprocessor()
        self._preprocessor.fit(X_tr)

        self._cat_feature_names = self._preprocessor\
            .named_transformers_["cat"]\
            .named_steps["onehot"]\
            .get_feature_names_out(input_features=self._columns_to_encode)
        
        self._new_feature_names = self._columns_to_scale + list(self._cat_feature_names)

        self.X_tr = self._preprocessor.transform(X_tr)
        self.X_te = self._preprocessor.transform(X_te)

        self.y_tr = y_tr.to_numpy()
        self.y_te = y_te.to_numpy()  

    def make_loaders(self): 
        """Create data loaders for train and test sets. 
        
        Details: only select positive class samples for train and test sets."""

        if self.X_tr is None or self.X_te is None or self.y_tr is None or self.y_te is None: 
            raise ValueError("Data must be preprocessed before creating data loaders.")

        idxs_tr, idxs_te = np.where(self.y_tr == 1)[0], np.where(self.y_te == 1)[0]
        X_tr, X_te, y_tr, y_te = self.X_tr[idxs_tr], self.X_te[idxs_te], self.y_tr[idxs_tr], self.y_te[idxs_te]

        datasets = dta.create_datasets(X_tr, y_tr, X_te, y_te)
        data = dta.DataBunch(*dta.create_loaders(datasets, bs=self.batch_size, device="cpu"))

        self.train_loader, self.test_loader = data.train_dl, data.test_dl

    def _inverse_scaling(self, x: np.ndarray) -> np.ndarray: 
        x_new = self._preprocessor\
            .named_transformers_["num"]\
            .named_steps["scaler"]\
            .inverse_transform(x)

        return x_new
    
    def _inverse_one_hot(self, x: np.ndarray) -> np.ndarray: 
        x_new = self._preprocessor\
            .named_transformers_["cat"]\
            .named_steps["onehot"]\
            .inverse_transform(x)
        
        return x_new

    def generate(self, new_samples: Union[torch.Tensor, np.ndarray]):
        """Make feature matrix and dataframe from synthetic samples."""

        if isinstance(new_samples, torch.Tensor):
            new_samples = new_samples.detach().numpy()

        numeric_cols_ixs = [ix for ix, col in enumerate(self._new_feature_names) if col in self.feature_types["numeric"]]
        category_cols_ixs = [ix for ix, col in enumerate(self._new_feature_names) if col not in self.feature_types["numeric"]]

        x_num = new_samples[:, numeric_cols_ixs]
        x_cat = new_samples[:, category_cols_ixs]

        self.X_syn = np.concatenate([x_num, x_cat], axis=1)

        x_num = self._inverse_scaling(x_num)
        x_cat = self._inverse_one_hot(x_cat)

        self.synthetic_df = pd.DataFrame(
            data=np.concatenate([x_num, x_cat], axis=1),
            columns=self.feature_types["numeric"] + self.feature_types["category"]
        ) 