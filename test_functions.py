""""
Module to test churn script functions using Pytest.

Author: Derrick
Date: November 2022
"""

import logging
import pickle
import pytest
import pandas as pd
from pandas.api.types import is_numeric_dtype
from functions.data import process_data
from functions.model import train_model
from functions.model import compute_model_metrics
from functions.model import inference

logging.basicConfig(
    filename='logs/tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture()
def dff():
    '''
    Pytest fixture to pass dataframe to various tests.
    '''
    dff_ = pd.read_csv("./data/census.csv")
    return dff_


@pytest.fixture()
def train_data(dff):
    '''
    Pytest fixture to pass train data to various tests.
    '''
    label = "salary"
    cat_features = dff.drop(label, axis=1).select_dtypes('object').columns
    x_train, y_train, encoder, labenc = process_data(
        dff,
        categorical_features=cat_features,
        label=label
        )
    return x_train, y_train


@pytest.fixture()
def model():
    """
    Pytest fixture to load model.
    """
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def test_process_data(dff):
    '''
    test data import - This tests the ability to load the training data.
    '''
    label = "salary"
    cat_features = dff.drop(label, axis=1).select_dtypes('object').columns
    try:
        x_train, y_train, encoder, labenc = process_data(
            dff,
            categorical_features=cat_features,
            label=label
            )
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data(): The file wasn't found: %s", err)
    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data(): The file was loaded but,\
         doesn't appear to have rows and columns: %s", err)
    try:
        assert y_train.shape[0] > 0
        assert len(y_train.shape) == 1
        assert is_numeric_dtype(y_train)
    except AssertionError as err:
        logging.error("Testing import_data(): The file was loaded, but\
         the labels were not encoded correctly: %s", err)


def test_train_model(train_data):
    '''
    test train_model - This tests the ability to train a model.
    '''
    x_train, y_train = train_data[0], train_data[1]
    model = train_model(x_train, y_train)
    assert model is not None
    assert model.oob_score_ is not None
    assert model.n_outputs_ == 1
    assert model.n_classes_ == 2
    logging.info("Testing train_model: SUCCESS")


def test_compute_model_metrics(train_data):
    '''
    test compute_model_metrics - This tests the ability to
    compute model metrics.
    '''
    x_train, y_train = train_data[0], train_data[1]
    model = train_model(x_train, y_train)
    y_preds = model.predict(x_train)
    precision, recall, fbeta = compute_model_metrics(y_train, y_preds)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None
    assert precision > 0
    assert recall > 0
    assert fbeta > 0
    assert precision <= 1
    assert recall <= 1
    assert fbeta <= 1
    logging.info("Testing compute_model_metrics: SUCCESS")


def test_inference(model, train_data):
    '''
    Pytest to test inference function.
    '''
    x_data, y_data = train_data[0], train_data[1]
    y_preds = inference(model, x_data)
    assert y_preds is not None
    assert len(y_preds) == len(y_data)
    logging.info("Testing inference: SUCCESS")
