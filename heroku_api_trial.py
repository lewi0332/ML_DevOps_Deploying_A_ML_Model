import pandas as pd
import requests
import json

df = pd.read_csv("data/census.csv")

test = df.drop('salary', axis=1).iloc[0].to_dict()

# Post request to the API.
response = requests.post("https://ml-devops-project-3.herokuapp.com/predict", data=json.dumps(test))
print(response.json()) # "<=50k"
