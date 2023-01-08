"""
This script contains the functions to train and test the model.

Author: Derrick Lewis
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    #TODO Optional: implement hyperparameter tuning.
    Trains a Random Forest Classifier model and returns it.

    Inputs
    ---
    X_train : np.array
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
    rfm.fit(X_train, y_train)
    return rfm


def compute_model_metrics(y, preds):
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
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ---
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    ---
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
