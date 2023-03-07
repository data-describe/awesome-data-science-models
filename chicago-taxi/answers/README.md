# Chicago Taxi
A wide and deep neural net implemented using the Keras Functional API to predict trip duration for Chicago taxi rides.

## Files
```
chicago-taxi
│   README.md
│   AI Platform.ipynb
│   hptuning_config.yaml
│   create_sample.py
│   calc_mape.py
│   requirements.txt
│   setup.py 
│   predictor.py
│   input_sample.json
│   x_scaler      
│
└───trainer
│   │   __init__.py
│   │   create_data_func.py
│   │   model.py
│   │   task.py
│   │   create_scaler_func.py
```
**AI Platform.ipynb** 
<br>
All of the commands necessary to tune, train, and host the neural network on AI Platform.

**hptuning_config.yaml**
<br>
Parameters to be used in hyperparameter tuning.

**create_sample.py**
<br>
A script that can be used to generate a sample model for inference.

**calc_mape.py**
<br>
A script that can be used to calculate the Mean Absolute Percengate Error of the model for a given sample size of the test set.

**requirements.txt**
<br>
Packages required in order to complete the Training in local execution mode.

**setup.py**
<br>
Specifies additional packages that are required for a training job submission to AI Platform.

**predictor.py**
<br>
Code necessary for implementation of the custom prediction routine.

**input_sample.json**
<br>
Sample preprocessed json object to be passed for inference

**x_scaler**
<br>
A MinMax scaler object that has been fit over the continuous features in the training set. 

**trainer/create_data_func.py**
<br>
Logic required to complete preprocessing and generate training, validation, and test datasets.  This step is callable as an attribute in trainer/task.py.

**trainer/create_scaler_func.py**
<br>
Logic required to train a MinMax scaler on the continuous features in the training set.  Similar to trainer/create_data_func.py, this step is callable as an attribute in trainer/task.py and needs to be completed only once for data preparation.

**trainer/model.py**
<br>
Includes logic necessary to build the wide and deep neural network using the Keras Functional API.

**trainer/task.py**
<br>
Main module responsible for accepting input parameters and executing the training job.

## Running Instructions
Instructions for running the process are outlined in AI Platform.ipynb.
