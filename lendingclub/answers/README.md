# Goal

The lending club example is an XGboost model used to demonstrate Google Cloud Platform's CloudML hypertune feature which is used to tune the model's parameters. The lending club model itself is a binary classification model which predicts those customers who have a high likelihood of defaulting on loans given a list of their personal and credit report attributes. XGboost has several tunable parameters. In this example we explore only three of them: max_depth, num_boost_round, and booster. You can pick any of the parameters however and select a range of values to search over to find the optimal combination to maximize a supplied metric.

# Directions

Use this notebook: "/sanofi-ml-workshop-repo/lendingclub/answers/LendingClub_xgboost.ipynb". It is set to run each cell one at a time.

Set environment variables in first cell if necessary in the first cell.

Part 1 of the notebook creates a python code file which builds an XGboost model. For the parser arguments, the three chosen parameters are set. You can change these to experiment with different ones from the list of parameters for XGboost.

The next part of the code reads in the raw input csv file and does some data cleaning, for example, filling in na values and capping outlier values.

Next, it one-hot-encodes categorical values and flags those as well as the numerical variables to create a set of predictors. Also, a training and testing time period is set, as well as the target variable.

An XGboost Classifer is created with the three parameters for which we want to do a grid search over values. We choose the AUC as the metric to optimize. All other parameters are set to their defaults.

Next create a hypertune instance from the CloudML Hypertune Module. 

Finally, we name the output model file and upload it to Google Cloud Storage.

In Part 2, we set up the range of values that the hypertune model uses to search through to find the combination that optimizes the AUC. We can set a range of values for max_depth and num_boost_rounds. We can set different types of boosters for the booster parameter.

In Part 3, we simply submit the files written previously to the AI Platform.

Finally, we can display the optimal parameter values and the AUC metric achieved with that parameter combination.

