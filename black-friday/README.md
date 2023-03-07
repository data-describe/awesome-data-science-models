# Product Category Suggestions for Black Friday dataset
Recommending product categories with multi-class categorization in AI Platform

## Files
```
black-friday
│   README.md
│   AI Platform.ipynb
│   generate_sample.py
│   requirements.txt
│   setup.py 
│   predictor.py
│   input_sample.json
│
└───trainer
│   │   __init__.py
│   │   create_data_func.py
│   │   hp_tuning.py
│   │   rf_trainer.py
```
**AI Platform.ipynb** 
<br>
All of the commands necessary to tune, train, and host the model on AI Platform.

**generate_sample.py**
<br>
A script that can be used to generate a sample for inference.

**requirements.txt**
<br>
Packages required in order to complete the training in local execution mode.

**setup.py**
<br>
Specifies additional packages that are required for a training job submission to AI Platform.

**predictor.py**
<br>
Code necessary for implementation of the custom prediction routine.

**input_sample.json**
<br>
Sample preprocessed json object to be passed for inference

**trainer/create_data_func.py**
<br>
Logic required to complete preprocessing and generate training and test datasets.  This module is callable as an attribute in trainer/rf_trainer.py.


**trainer/hp_tuning.py**
<br>
Includes logic necessary to perform hyperparameter tuning on the Random Forest model.  Similar to trainer/create_data_func.py, this module is callable as an attribute in trainer/rf_trainer.py.

**trainer/rf_trainer.py**
<br>
Main module responsible for accepting input parameters and executing the training job.

## Running Instructions
Instructions for running the process are outlined in AI Platform.ipynb.