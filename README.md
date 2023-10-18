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