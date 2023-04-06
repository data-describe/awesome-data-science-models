
# Awesome Data Science Models

The goal of these projects is to provide tutorials (and answers) of our most popular models we use for training.

They all use [data⎰describe](https://github.com/data-describe/data-describe) our open source accelerator for EDA.

| Library   Item | Title | Description | Audience |
|-|-|-|-|
| 101 | Baseball Pitch Predictor | Kubeflow pipeline for modeling   around pitch type | Citizen Data Scientists,   Advanced Data Scientists |
| 102 | Beatles Like Predictor | Predict who likes the Beatles   using purely GCP AutoML Tables | Citizen Data Scientists |
| 201 | Lending Club bad-loan | predict if a loan applicant is   high risk using XGBoost and hyper-parameter training. | Data Scientists, Data Engineers |
| 301 | Census Income | The census income example is a   logistic regression model used to demonstrate Google Cloud Platform's AI   Platform.  | Data Scientists, Data Engineers |
| 202 | Taxi Cab Prediction | A wide and deep neural net   implemented using Tensorflow predict trip duration for Chicago taxi rides. | Data Scientists, Data Engineers |
| 203 | Black Friday | Recommending product categories   with multi-class categorization in AI Platform with custom prediction   routines, custom hyperparameter tuning | Data Scientists, Data Engineers |
| 302 | Cellular Imaging | Recursion Cellular Image   Classification with AI-Platform with TF2, TPU, and advanced Engineering | Data Scientists, Data Engineers,   HCLS |
| 401 | NASA IOT Signal processing  | Processing Streaming data with signal windowing and live prediction using AI Platform and GCP | MLOps, Advanced Data Scientists |
| 402 | Bearing Condition Monitoring | Analyzing the Vibrations and Motor current of Bearings to predict Remaining Useful LifeCycle (RUL) using Vertex AI | Data Scientist, Data Engineers |

## install instructions

These examples are meant to run on GCP AI Platform. They may very well run elsewhere but we haven't tested.


Create an instance for AI Platform notebooks:

1. Choose Tensorflow Enterprise 2.1 (No GPUs)
    - Make sure you have 4 CPUs and at least 15 GB of memory
    - Click Open JupyterLab
    - Use the Launcher (right-hand-side of screen) to open a Terminal...
2. Install data describe:
     - pip install data-describe[all]
     - pip install xgboost==0.82
     - pip install pandas_gbq
     - pip install google-cloud-bigquery
     - pip install google-cloud-storage



Clone the examples:
```
git clone https://github.com/data-describe/awesome-data-science-models.git
```


## Sources of Data

### Beatles Like Predictor
The original dataset is available from listenbrainz and provided by BigQuery [here](https://console.cloud.google.com/bigquery?project=listenbrainz&page=table&t=listen&d=listenbrainz&p=listenbrainz&redirect_from_classic=true)

### Black Friday
The original dataset is on Kaggle, and can be found [here](https://www.kaggle.com/sdolezel/black-friday).

### Census Income
The original dataset is on the UCI Machine Learning Repository, and can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).

### Chicago Taxi
The original dataset is a BigQuery public dataset, and can be found [here](https://console.cloud.google.com/marketplace/product/city-of-chicago-public-data/chicago-taxi-trips?filter=solution-type:dataset&id=13c38348-0610-4185-a8f7-b5add142fcbe&project=mwpmltr&folder=&organizationId=). More information on BigQuery public datasets can be found [here](https://cloud.google.com/bigquery/public-data).

### Lending Club
The original dataset is made public by LendingClub, and can be found at their website or [here](https://www.kaggle.com/wordsforthewise/lending-club). The dataset used for this demo is a subset of the original dataset.

### Cellular Image
The original data is part of the tensorflow datasets. We have copied this data into a public bucket for the demo [here](https://console.cloud.google.com/storage/browser/temp_data_bukcet).

### Nasa IOT Data
J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA

### Bearing Condition Monitoring
The original data is a public dataset of the Chair of Design and Drive Technology, Paderborn University. It is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License and it is available for downloading it [here](https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download)
