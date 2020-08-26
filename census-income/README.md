# Goal

The census income example is a logistic regression model used to demonstrate Google Cloud Platform's AI Platform. In the exercise, it is demonstrated how to set up your environment, bring in your data, create a model training folder, submit a local job on AI Platform (for debugging purposes) and predict from that model, submit a training job to AI Platform, upload the model, create a model resource and version on AI Platform, and make a prediction by one of two ways: gcloud command line and Python.

The objective of the model itself is to predict if a said person, given their attributes, earns more than $50k per annum or not.

## Running Instructions
Instructions for running the process are outlined in AI Platform.ipynb.

## Files
```
├── census-income
│   ├── answers
│   │   ├── AI Platform.ipynb
│   │   └── trainer
│   │       ├── __init__.py
│   │       └── task.py
│   ├── exercises
│   │   └── README.md
│   ├── notebooks
│   │   └── EDA_Census_Data.ipynb
│   └── README.md
```

**AI Platform.ipynb**
<br>
All of the commands necessary to tune, train, and host the model on AI Platform. This is a step-by-step notebook with explicit instructions on how to carry out the tasks outlined in the "Goals" section above.

**requirements.txt**
<br>
Packages required in order to complete the training in local execution mode.

**trainer/task.py**
<br>
Main module responsible for accepting input parameters and executing the training job.

**notebooks/EDA_Census_Data.ipynb**
<br>
Notebook which runs an Exploratory Data Analysis on the dataset.