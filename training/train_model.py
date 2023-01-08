"""
Script to train machine learning model.

Author: Derrick Lewis
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from functions.data import process_data
from functions.model import train_model

# Add code to load in the data.
df = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test
# split.
train, test = train_test_split(df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
x_train, y_train, encoder, labenc = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model
model = train_model(x_train, y_train)

# Save model
with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("../model/labenc.pkl", "wb") as f:
    pickle.dump(labenc, f)
