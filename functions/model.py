"""
This script contains the functions to train and test the model.

Author: Derrick Lewis
"""
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(
        x_train: np.array,
        y_train: np.array) -> RandomForestClassifier:
    """
    #TODO Optional: implement hyperparameter tuning.
    Trains a Random Forest Classifier model and returns it.

    Inputs
    ---
    x_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    ---
    model
        SciKit-Learn RandomForestClassifier machine learning model object.
    """
    rfm = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        max_features="sqrt",
        min_samples_leaf=50
        )
    rfm.fit(x_train, y_train)
    return rfm


def compute_model_metrics(
        y_test: np.array,
        preds: np.array) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ---
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ----
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y_test, preds, beta=1, zero_division=1)
    precision = precision_score(y_test, preds, zero_division=1)
    recall = recall_score(y_test, preds, zero_division=1)
    return precision.round(3), recall.round(3), fbeta.round(3)


def compare_slice_performance(
        dff: pd.DataFrame,
        d_slice: str,
        model: RandomForestClassifier,
        x_test: np.array,
        y_test: np.array):

    """
    Compares the performance of the model on slices of the data.

    Inputs
    ---
    dff : pd.DataFrame
        Dataframe containing the features and label.
    slice : str
        Name of the column to slice the data on.
    model : sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    x_test : np.array
        Test data after process_data.
    y_test : np.array
        Test labels after process_data.
    Returns
    ----
    precision : float
    recall : float
    fbeta : float
    """

    for cat_feat in dff[d_slice].unique():
        x_data = x_test[dff[d_slice] == cat_feat]
        y_data = y_test[dff[d_slice] == cat_feat]
        y_preds = inference(model, x_data)
        precision, recall, fbeta = compute_model_metrics(y_data, y_preds)
        print(f"Performance for slice where {d_slice} is {cat_feat} -\
Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")


def inference(
        model: RandomForestClassifier,
        x_data: np.array) -> np.array:
    """ Run model inferences and return the predictions.

    Inputs
    ---
    model : ???
        Trained machine learning model.
    x_data : np.array
        Data used for prediction.
    Returns
    ---
    preds : np.array
        Predictions from the model.
    """
    return model.predict(x_data)
