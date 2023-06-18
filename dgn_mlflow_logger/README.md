# MLFlow logger (ElectroCardioGuard)
- using mlflow with remote tracking and artifact storage


1. Install using 
```
pip install -e .
```

2. Edit the `.env` file next to `dgn_mlflow_logger/logger.py` to contain the settings of your MLFlow tracking server. To setup the server, please refer to https://mlflow.org/docs/latest/tracking.html. 

Required variables:

**MLFLOW_TRACKING_URI**: URI of your tracking server (including port)

**MLFLOW_TRACKING_USERNAME**, **MLFLOW_TRACKING_PASSWORD**: Mlflow authentication

**MLFLOW_S3_ENDPOINT_URL**: URI of your S3 storage (including port). For this purpose, we use minio (https://min.io/).

**AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY**: S3 authentication
