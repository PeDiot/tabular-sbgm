from pandas.core.frame import DataFrame
from typing import Dict

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
            self.reference = cfg["reference"]
        
        self.column_types = {}

        for col in cfg["numeric"]:
            self.column_types[col] = "numeric"
        for col in cfg["category"]:
            self.column_types[col] = "category" 

        self._feature_types = {
            feature: type for feature, type in self.column_types.items() if feature!=cfg["target"]
        }

        self.train_loader, self.test_loader = None, None

    def __repr__(self) -> str:
        if self.classification: 
            return f"Data(target={self.target}, classification={self.classification}, reference={self.reference}, test_prop={self.test_prop}, batch_size={self.batch_size})"
        else:
            return f"Data(target={self.target}, classification={self.classification}, test_prop={self.test_prop}, batch_size={self.batch_size})"

    def _make_preprocessor(self): 

        self._columns_to_scale = [
            key for key, type in self._feature_types.items() if type=="numeric"
        ]
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        self._columns_to_encode = [
            key for key, type in self._feature_types.items() if type=="category"
        ]
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

        self.X = df.drop(columns=[self.target])

        if self.classification: 
            self.y = df[self.target].apply(lambda x: 1 if x==self.reference else 0)

        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size=self.test_prop, random_state=42)

        self._make_preprocessor()
        self._preprocessor.fit(X_tr)

        self._cat_feature_names = self._preprocessor\
            .named_transformers_["cat"]\
            .named_steps["onehot"]\
            .get_feature_names_out(input_features=self._columns_to_encode)
        
        self.feature_names = self._columns_to_scale + list(self._cat_feature_names)

        self._X_tr = self._preprocessor.transform(X_tr)
        self._X_te = self._preprocessor.transform(X_te)

        self._y_tr = y_tr.to_numpy()
        self._y_te = y_te.to_numpy()  

        datasets = dta.create_datasets(self._X_tr, self._y_tr, self._X_te, self._y_te)
        data = dta.DataBunch(*dta.create_loaders(datasets, bs=32, device="cpu"))

        self.train_loader, self.test_loader = data.train_dl, data.test_dl