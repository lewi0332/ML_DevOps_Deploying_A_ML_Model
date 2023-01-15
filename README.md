[![example workflow](https://github.com/lewi0332/ML_DevOps_Deploying_a_Machine_Learning_Model/actions/workflows/pylint.yml/badge.svg)](https://github.com/lewi0332/ML_DevOps_Deploying_a_Machine_Learning_Model/actions)
[![example workflow](https://github.com/lewi0332/ML_DevOps_Deploying_a_Machine_Learning_Model/actions/workflows/python-package.yml/badge.svg)](https://github.com/lewi0332/ML_DevOps_Deploying_a_Machine_Learning_Model/actions)

# Machine Learning DevOps<br> - Deploying a Machine Learning Model

Using Continuos Integraion with Github Actions, Data Version control with DVC, API creation with FastAPI and 
Continuous Deployment with Heroku.

---
Module 3 Project for the [Udacity Machine Learning DevOps Engineer NanoDegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)

To view the resulting API:
https://ml-devops-project-3.herokuapp.com/docs

---
## Goal 

This project applies the skills acquired in the course to develop a classification model on publicly available Census Bureau data. Then create unit tests to monitor the model performance on various slices of the data. Then, deploys a model using the FastAPI package and create API tests to monitor the system. Both the model validation and the API tests are incorporated into a CI/CD framework using GitHub Actions.

Two datasets were provided in the Udacity starter code and an outline of a basic classification model.

The model was loosely trained on the data to simulate a typical Data Science project, however the focus of this repository is to create a reusable and production ready deployment of a model. Therefore, the performance of the predictions should be excused and certainly, not used. 

## Components

The following pieces were set up and explored in this project

**Repositories**

- Created a directory for the project and initialized Git.

**GitHub Actions**

- Setup GitHub Actions on the repository.
  - Created a workflow to run `pylint` on each "Push" or "Pull" to the `main` branch
  - Created a workflow to run `pytest` and `flake8` on each "Push" or "Pull" to the main branch
    - Configured github utilize Data Version Control data stored in a Google Cloud Project bucket during testing.  

**Data**

- Using DVC, set up and tracked data versioning of the [UCI Machine Learning Census Dataset](https://archive.ics.uci.edu/ml/datasets/census+income) supplied by Udacity.  Changes were stored in a GCP Cloud-Storage Bucket.
  - `/functions/data.py:process_data` - Main function to clean and prepare data for modeling. 

**Model**

- Wrote a machine learning model that trains on the clean data and saves the model components to Cloud storage. 
  - `/functions/model.py` - Core functions of model training and scoring
  - `/train_model.py` - Main script to train the data using functions from the previous file.
- Set up `DVC` to version model training artifacts along with data to be reused.
- Wrote unit tests for most functions in the model code.
  - `/test_functions.py` - Main test suite for modeling functions. This test is checked on each push to the main Github branch via Github Actions. 
- Wrote a function that outputs the performance of the model on slices of the data.
  - Using a categorical feature, the model can be evaluated on each value in isolation to look for bias
- Wrote a model card using the provided template.

  
**API Creation**

- Created a RESTful API using FastAPI
  - `main.py` - API set up and functions 
    - `/` - The root page uses a `GET` to return a welcome message
    - `/sample_data/` - POST method endpoint returns a schema with examples of how data can be submitted for predictions
    - `/check_data/` - POST method endpoint returns the data that was submitted in the api call to verify and debug any attempts. 
    - `/predict/` - POST method endpoint to make a prediction on the data. 

- Using Pydantic, created a data model struction that includes type hints. This allows for automatic creation of API documentation with examples and data types. 

- Wrote Unit tests to verify future changes to the `main.py` API script work as intended. 
  - `test_api.py` - Main script to test API. Configured to be tested by Github Actions. 


**API Deployment to a Cloud Application Platform**

- Created a Heroku App which deploys from this GitHub repository as long as all Github actions are passing. Thus, a continuous deployment environment.
  - `Procfile` - Heroku dyno creation
- Configure Heroku build to include Aptfile to install and configure the environment to safely handle Google Cloud Credentials using the [GCP Auth Buildpack](https://elements.heroku.com/buildpacks/buyersight/heroku-google-application-credentials-buildpack) and . Thereby, this app can communicate with cloud storage to interact with all `DVC` stored objects. 
  - Heroku app running at https://ml-devops-project-3.herokuapp.com/
- Wrote a script that uses the requests module to do one POST on the live API.
  - `heroku_api_trial.py` - Reads data in to send a sample individual to make an income prediction on. 


## Installation

1. Install necessary libraries using Anaconda
    - `conda create -n proj3 python=3.8.15`
    - `conda install --name proj3 --file requirements.txt`
  - 
1. Activate the Envirnoment
    - `conda activate proj3`

## Author

-   **Derrick Lewis**  [Portfolio Site](https://www.derrickjameslewis.com) - [linkedin](https://www.linkedin.com/in/derrickjlewis/)


## License

This project is licensed under Udacity Educational License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

I would like to thank [Udacity](https://eu.udacity.com/) for this amazing project.