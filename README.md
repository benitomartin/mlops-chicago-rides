# MLOps Project Chicago Taxi Prices Prediction

<p>
    <img src="/images/taxi.jpg"/>
    </p>

This is a personal MLOps project based on a [BigQuery](https://console.cloud.google.com/marketplace/product/city-of-chicago-public-data/chicago-taxi-trips?project=taxifare-mlops) dataset for taxi ride prices in Chicago.

Below you can find some instructions to understand the project content. Feel free to clone this repo 😉


## Tech Stack

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23d9ead3.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=MLflow&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Grafana](https://img.shields.io/badge/grafana-%23F46800.svg?style=for-the-badge&logo=grafana&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Project Structure

The project has been structured with the following folders and files:

- `.github:` contains the CI/CD files (GitHub Actions)
- `data:` dataset and test sample for testing the model
- `integration_tests:` prediction integration test with docker-compose
- `lambda:` test of the lambda handler with and w/o docker
- `model:` full pipeline from preprocessing to prediction and monitoring using MLflow, Prefect, Grafana, Adminer, and docker-compose
- `notebooks:` EDA and Modeling performed at the beginning of the project to establish a baseline
- `tests:` unit tests
- `terraform:` IaC stream-based pipeline infrastructure in AWS using Terraform
- `Makefile:` set of execution tasks
- `pyproject.toml:` linting and formatting
- `setup.py:` project installation module
- `requirements.txt:` project requirements

## Project Description

The dataset was obtained from Kaggle and contains various columns with car details and prices. To prepare the data for modeling, an **Exploratory Data Analysis** was conducted to preprocess numerical and categorical features, and suitable scalers and encoders were chosen for the preprocessing pipeline. Subsequently, a **GridSearch** was performed to select the best regression models, with RandomForestRegressor and GradientBoostingRegressor being the top performers, achieving an R2 value of approximately 0.9.






 
 
 
 MLFlow must be run in interface folder

To Create the DB
 mlflow ui --backend-store-uri sqlite:///mlflow.db

To run the code and log data
 python .\main.py


conda install -c conda-forge tensorflow


Prefect:

 prefect cloud login

 or locally

 prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api


 uvicorn src.api.fast:app --reload


 docker build --no-cache -t ride-prediction .


 docker run -it -e PORT=8000 -p 8000:8000 ride-prediction

Api from dockerfile runs here
 http://127.0.0.1:8000


 docker run -it --rm -p 8000:8000 -e MODEL_TARGET= local -e LOCAL_REGISTRY_PATH=C:\Users\bmart\OneDrive\11_MLOps\taxifare\training_outputs\models aaa

C:\Users\bmart\OneDrive\11_MLOps\taxifare
docker run -it --rm -p 8000:8000  --env-file C:\Users\bmart\OneDrive\11_MLOps\taxifare\.env:/app/.env aaa


$env:PYTHONPATH += "C:\Users\bmart\OneDrive\11_MLOps\taxifare"

$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\bmart\.google\credentials\taxifare-mlops-8bb6b290e1b8.json"

gcloud auth activate-service-account --key-file=$env:GOOGLE_APPLICATION_CREDENTIALS
gcloud auth list
gcloud auth login
gcloud auth application-default print-access-token
gcloud auth configure-docker
gcloud auth application-default login


C:\Users\bmart\AppData\Roaming\gcloud\application_default_credentials.json

docker run -it --rm -p 8000:8000 -v C:\Users\bmart\OneDrive\11_MLOps\taxifare\.env:/app/.env credentials

docker run -it --rm -p 8000:8000 -v C:\Users\bmart\OneDrive\11_MLOps\taxifare\.env:/app/.env -v C:\Users\bmart\.google\credentials:/root/.google/credentials gcs


docker run -e PORT=8000 -p 8000:8000 --env-file C:\Users\bmart\OneDrive\11_MLOps\taxifare\.env final:dev


 docker build -t eu.gcr.io/taxifare-mlops/taxi_image:prod .
 
 docker run -e PORT=8000 -p 8000:8000 --env-file .env eu.gcr.io/taxifare-mlops/taxi_image:prod

docker push eu.gcr.io/taxifare-mlops/taxi_image:prod



gcloud run deploy --image eu.gcr.io/taxifare-mlops/taxi_image:prod --memory=2Gi --region europe-west1 --env-vars-file .env.yaml


docker run -e PORT=8000 -p 8000:8000 --env-file C:\Users\bmart\OneDrive\11_MLOps\taxifare\.env model

gcloud auth activate-service-account taxifare-chicago@taxifare-mlops.iam.gserviceaccount.com --key-file=C:\Users\bmart\.google\credentials\taxifare-mlops-8bb6b290e1b8.json --project=taxifare-mlops