import requests
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture()
def data_low():
    df = pd.read_csv("data/census.csv")
    test = df.drop('salary', axis=1).iloc[0].to_dict()
    return test
@pytest.fixture()
def data_high():
    df = pd.read_csv("data/census.csv")
    test = df.drop('salary', axis=1).iloc[9].to_dict()
    return test


def test_get_data():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to the MLops Project 3 API! \
Please use docs to see the API documentation."}

def test_post_data_success(data_low):
    data = json.dumps(data_low)
    r = client.post("/predict/", data=data)
    assert r.status_code == 200


def test_post_data_fail():
    data = {"age": -5, "feature_2": "test string"}
    r = client.post("/predict/", data=json.dumps(data))
    assert r.status_code == 422


def test_post_low(data_low):
    data = json.dumps(data_low)
    r = client.post("/predict/", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == '<=50K'

def test_post_high(data_high):
    data = json.dumps(data_high)
    r = client.post("/predict/", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == '>50K'
