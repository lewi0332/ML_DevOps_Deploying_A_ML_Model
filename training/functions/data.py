"""
This script contains the functions to process the data.

Author: Udacity and Derrick Lewis
"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    dff,
    categorical_features=None,
    label=None,
    training=True,
    encoder=None,
    lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and
    a label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.

    Inputs
    ------
    dff : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be
        returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    x_data : np.array
        Processed data.
    y_data : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed in.
    """

    if label is not None:
        y_data = dff[label]
        x_data = dff.drop([label], axis=1)
    else:
        y_data = np.array([])

    x_categorical = x_data[categorical_features].values
    x_continuous = x_data.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        x_categorical = encoder.fit_transform(x_categorical)
        y_data = lb.fit_transform(y_data.values).ravel()
    else:
        x_categorical = encoder.transform(x_categorical)
        try:
            y_data = lb.transform(y_data.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    x_data = np.concatenate([x_continuous, x_categorical], axis=1)
    return x_data, y_data, encoder, lb
