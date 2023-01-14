# Put the code for your API here.
import numpy as np
import pandas as pd
import pickle
import json
from typing import Union
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI
from functions.model import inference
from functions.data import process_data
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Instantiate the app.
app = FastAPI(
    title="MLops Project 3 API",
    description="An API that returns a prediction of our classification model.",
    version="1.0.0",
)

#TODO load the model here

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("model/labenc.pkl", "rb") as f:
    labenc = pickle.load(f)

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    age: int = 38
    workclass: str = 'State-gov'
    fnlgt: int = 77516
    education: str = 'Bachelors'
    education_num: int = Field(13, alias='education-num')
    marital_status: str = Field('Never-married', alias='marital-status')
    occupation: str = 'Adm-clerical'
    relationship: str = 'Not-in-family'
    race: str = 'White'
    sex: str='Male'
    capital_gain: int=Field(2174, alias='capital-gain')
    capital_loss: int=Field(0, alias='capital-loss')
    hours_per_week: int=Field(40, alias='hours-per-week')
    native_country: str=Field('United-States', alias='native-country')

    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('age must be a positive integer')
        return v

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to the MLops Project 3 API! Please use /docs to see the API documentation."}


# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/sample_data/")
async def check_data(item: TaggedItem):
    """
    Use this to return the schema of the data needed for the API.
    """
    return item.schema_json(indent=2)

@app.post("/check_data/")
async def check_data(item: TaggedItem):
    return item

@app.post("/predict/")
async def get_items(inference_json: TaggedItem): 
    # Convert to pandas dataframe as expected by process_data
    inference_json = {k:[v] for k,v in inference_json}
    inference_json = pd.DataFrame(inference_json)
    # Undo the aliasing of the columns needed by pydantic.
    # (Why not just send a json?)
    inference_json.columns = [c.replace("_", "-") for c in inference_json.columns]
    data = process_data(
        dff=inference_json,
        categorical_features=categorical_features,
        # label='salary',
        training=False,
        encoder=encoder,
        labenc=labenc
        )

    prediction = inference(model, data[0])
    print(prediction)
    named_pred = labenc.inverse_transform(prediction)
    return named_pred[0]

