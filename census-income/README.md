# Goal
The census income example is a binary classification problem which has been used here to demonstrate the capabilities of Vertex AI, Google Cloud Platform's managed service for managing the entire end-to-end workflow of a machine learning use-case. In the exercise, it is shown:
- how to set up your environment
- bring in your custom data
- prepare data acc. to chosen Vertex AI service (e.g. AutoML Tabular classification)
- build a custom container on Artifact Registry and and use it on Vertex AI
- submit a hptuning and training jobs on Vertex AI
- do batch predictions on the trained model and get model explanations
- deploy to an endpoint for online prediction

The objective of the model itself is to predict whether a said person, given their attributes, earns more than $50k per annum or not.

## Running Instructions
The notebooks were run inside a Vertex AI user-managed workbench. The required dependencies, where required, are taken are of using '!pip install' commands inside that notebook itself, but make sure to restart the notebook kernel in that case.

## Files
```
├── EDA
│   └── Census_Income_EDA.ipynb
├── README.md
├── answers
│   ├── AutoML_batch_online_prediction_explain.ipynb
│   ├── hptuning
│   │   ├── Dockerfile
│   │   ├── hptuning.ipynb
│   │   └── trainer
│   │       ├── __init__.py
│   │       └── task.py
│   └── local_training.ipynb
├── exercises
│   ├── AutoML_batch_online_prediction_explain.ipynb
│   ├── hptuning
│   │   ├── Dockerfile
│   │   ├── hptuning.ipynb
│   │   └── trainer
│   │       ├── __init__.py
│   │       └── task.py
│   └── local_training.ipynb
```
**EDA/Census_Income_EDA.ipynb**
<br>
Notebook which runs an Exploratory Data Analysis on the dataset.

**exercises folder**
<br>
Contains all the tasks mentioned above, with guidance at every step on how to proceed.

**answers folder**
<br>
Contains working code snippets for all the tasks mentioned above. Preferably to be used only for reference when stuck, or comparing later on.