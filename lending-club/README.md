# Goal

The lending club example is an XGBoost model used to demonstrate Google Cloud Platform's CloudML hypertune feature which is used to tune the model's parameters. The lending club model itself is a binary classification model which predicts those customers who have a high likelihood of defaulting on loans given a list of their personal and credit report attributes. XGboost has several tunable parameters. In this example we explore only three of them: max_depth, num_boost_round, and booster. You can pick any of the parameters; however, and select a range of values to search over to find the optimal combination to maximize a supplied metric.

## Running Instructions
Instructions for running the process are outlined in Lending_Club.ipynb.

## Files
```
├── lending-club
│   ├── answers
│   │   ├── Lending_Club.ipynb
│   │   └── hptuning_config.yaml
│   │   └── trainer
│   │       ├── __init__.py
│   │       └── task.py
│   ├── EDA
│   │   ├── Lending_Club_EDA.ipynb
│   └── README.md
```

**GCP/answers/Lending_Club.ipynb**
<br>
All of the commands necessary to tune, train, and host the model on AI Platform. This is a step-by-step notebook with explicit instructions on how to carry out the tasks outlined in the "Goals" section above.

**GCP/answers/trainer/hptuning_config.yaml**
<br>
The YAML file which contains the information required for HP tuning.

**GCP/answers/trainer/task.py**
<br>
Main module responsible for accepting input parameters and executing the training job.

**GCP/EDA/Lending_Club_EDA.ipynb**
<br>
Notebook which runs an Exploratory Data Analysis on the dataset.

