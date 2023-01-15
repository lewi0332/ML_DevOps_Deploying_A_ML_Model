"""
This script is the main script for the prediction API.


Author: Derrick Lewis
"""
# Put the code for your API here.
import pickle
import os
import pandas as pd
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI
from functions.model import inference
from functions.data import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull --force") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Instantiate the app.
app = FastAPI(
    title="MLops Project 3 API",
    description="An API to return a prediction of our classification model.",
    version="1.0.0",
)

# Attemped @app.on_event("startup"), but this doesn't work with pytest.
# load the model and encoders the old fashioned way.
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
    """
    Pydantic model for the data to be sent to the API.
    Based on Census Income Data Set from UCI Machine Learning Repository.
    """
    age: int = 38
    workclass: str = 'State-gov'
    fnlgt: int = 77516
    education: str = 'Bachelors'
    education_num: int = Field(13, alias='education-num')
    marital_status: str = Field('Never-married', alias='marital-status')
    occupation: str = 'Adm-clerical'
    relationship: str = 'Not-in-family'
    race: str = 'White'
    sex: str = 'Male'
    capital_gain: int = Field(2174, alias='capital-gain')
    capital_loss: int = Field(0, alias='capital-loss')
    hours_per_week: int = Field(40, alias='hours-per-week')
    native_country: str = Field('United-States', alias='native-country')

    @validator('age')
    def age_must_be_positive(cls, v):
        """
        Ensure that the age is a positive integer.
        """
        if v < 0:
            raise ValueError('age must be a positive integer')
        return v


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    """
    This is the main entry for the API.
    """
    return {"greeting": "Welcome to the MLops Project 3 API! Please use \
docs to see the API documentation."}


# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/sample_data/")
async def sample_data(item: TaggedItem):
    """
    Use this to return the schema of the data needed for the API.
    """
    return item.schema_json(indent=2)


@app.post("/check_data/")
async def check_data(item: TaggedItem):
    """
    USe this to check the data you are sending to the API.
    """
    return item


@app.post("/predict/")
async def get_items(inference_json: TaggedItem):
    """
    Inference endpoint for the API.

    This is the endpoint that will be used to make predictions.
    Send a json with the data to be predicted. The json should
    be formatted per the shema in the endpoint /sample_data.
    """
    # Convert to pandas dataframe as expected by process_data
    inference_json = {k: [v] for k, v in inference_json}
    inference_json = pd.DataFrame(inference_json)
    # Undo the aliasing of the columns needed by pydantic.
    # (Why not just send a json?)
    inference_json.columns = [
        c.replace("_", "-") for c in inference_json.columns
        ]
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
