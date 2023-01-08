"""
Script to train machine learning model.

Author: Derrick Lewis
"""
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model, compare_slice_performance, compute_model_metrics

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
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Train and save a model
model = train_model(x_train, y_train)


# Compare the performance of the model on the test set

x_test, y_test, encoder, labenc = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    labenc=labenc
)

y_preds = model.predict(x_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
print(f"Performance for slice {cat_feat} - Precision: {precision},\
 Recall: {recall}, Fbeta: {fbeta}")

original = sys.stdout
with open("../logs/slice_output.txt", "w") as f:
    sys.stdout = f
    print(compare_slice_performance(
            test,
            'education',
            model=model,
            x_test=x_test,
            y_test=y_test))
    sys.stdout = original

# Save model
with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("../model/labenc.pkl", "wb") as f:
    pickle.dump(labenc, f)

