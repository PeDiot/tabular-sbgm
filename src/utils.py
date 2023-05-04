from typing import Dict, List

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report

import yaml
from yaml import Loader
import pickle as pkl 

from rich.table import Table
import rich

from sdv.metadata import SingleTableMetadata
from src.data import Data


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pkl.dump(obj, f)
        print(f"Saved pickle object to {path}") 

def load_pickle(path: str):
    with open(path, "rb") as f:
        obj = pkl.load(f)
    return obj

def load_config(cfg_path: str) -> Dict:
    """Load YAML config file."""

    with open(cfg_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader)

    return cfg

def save_config(cfg: Dict, path: str) -> None:
    """Save YAML config file."""

    with open(path, "w") as ymlfile:
        yaml.dump(cfg, ymlfile)

def check_input_dim(model_cfg: Dict, input_dim: int) -> Dict:
    """Check input dimension entered in config.
    
    Args:
        model_cfg (Dict): Model configuration dictionary.
        input_dim (int): Input dimension.
        
    Returns:
        model_cfg (Dict): Model configuration dictionary with corrected input dimension."""

    if not (model_cfg["input_layer"][0][1] == model_cfg["output_layer"][0][-1] == input_dim):
        model_cfg["input_layer"][0][1] = model_cfg["output_layer"][0][-1] = input_dim

    return model_cfg

def get_clf_metrics(
    clf: BaseEstimator, 
    X_tr: np.ndarray, 
    y_tr: np.ndarray, 
    X_te: np.ndarray, 
    y_te: np.ndarray, 
    target_names: list
) -> Dict:
    """Get precision, recall, f1-score and accuracy for train and test sets."""

    y_tr_pred = clf.predict(X_tr)
    y_te_pred = clf.predict(X_te)
    
    metrics = {
        "train": classification_report(y_tr, y_tr_pred, target_names=target_names, output_dict=True),
        "test": classification_report(y_te, y_te_pred, target_names=target_names, output_dict=True)
    }
    
    return metrics

def display_clf_metrics(clf_report: Dict, clf_aug_report: Dict, digits: int=3, title: str="Classification Report"): 
    """Display classification metrics for original and augmented classifiers.
    
    Args:
        clf_report (Dict): Classification report for original classifier.
        clf_aug_report (Dict): Classification report for augmented classifier.
        digits (int, optional): Number of digits to display. Defaults to 3."""

    for ds_type in ("train", "test"): 
        table = Table(title=f"{title} ({ds_type})")

        table.add_column("Metric")
        table.add_column("Original Data")
        table.add_column("Augmented Data")

        for metric, value in clf_report[ds_type].items():
            if metric not in ["macro avg", "weighted avg"]:

                if isinstance(value, dict):
                    for k, v in value.items():
                        if k != "support":
                            v_aug = clf_aug_report[ds_type][metric][k]
                            table.add_row(f"{metric} {k}", str(round(v, digits)), str(round(v_aug, digits))) 

                else:
                    value_aug = clf_aug_report[ds_type][metric]
                    table.add_row(metric, str(round(value, digits)), str(round(value, digits))) 
                    
        rich.print(table) 

def make_sdv_metadata(data: Data, add_target: bool=False) -> SingleTableMetadata:
    """Make metadata from data for SDV.
    
    Args:
        data (Data): Data object.
        add_target (bool, optional): Whether to add target variable to metadata. Defaults to False.
        
    Returns:
        metadata (SingleTableMetadata): Metadata dictionary."""

    metadata = SingleTableMetadata()

    for var in data.feature_types["category"]: 
        metadata.add_column(column_name=var, sdtype="categorical")

    for var in data.feature_types["numeric"]: 
        metadata.add_column(column_name=var, sdtype="numerical", computer_representation="Float")

    if add_target:
        if data.classification: 
            metadata.add_column(column_name=data.target, sdtype="categorical")
        else: 
            metadata.add_column(column_name=data.target, sdtype="numerical", computer_representation="Float")

    return metadata 

