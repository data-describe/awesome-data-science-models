# Product Category Suggestions for Black Friday dataset
Recommending product categories with multi-class categorization in Vertex AI

## Files
```
black-friday
│   README.md
└───EDA
│   │   EDA.ipynb
└───answers/Vertex AI
│   │   Vertex AI.ipynb
│   │   Dockerfile
|   |   requirements.txt
│   │   app.py
│   │   utils.py
```
**Vertex AI.ipynb** 
<br>
All of the commands necessary to tune, train, and host the model on Vertex AI.

**Dockerfile**
<br>
A Dockerfile to create a container for hosting the custom prediction routine.

**requirements.txt**
<br>
Packages required in order to complete the training in local execution mode.

**app.py**
<br>
The flask API backend code for post processing of predictions.

**utils.py**
<br>
Necessary code for processing data, training/tuning models, and feature selection.

**EDA.ipynb**
<br>
Notebook for exploratory data analysis.

## Running Instructions
Instructions for running the process are outlined in Vertex AI.ipynb.