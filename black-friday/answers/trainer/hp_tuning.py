from google.cloud import storage
import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def complete_hp_tuning(x_train_part, y_train_part, project_id, bucket_name, num_iterations):
    # perform hyperparameter tuning
    best_accuracy = -1
    for i in range(0, num_iterations):
        # ramdom split for train and validation
        x_train, x_test, y_train, y_test = train_test_split(x_train_part, y_train_part, test_size=0.2)

        # randomly assign hyperparameters
        n_estimators = np.random.randint(10, 1000)
        max_depth = np.random.randint(10, 1000)
        min_samples_split = np.random.randint(2, 10)
        min_samples_leaf = np.random.randint(1, 10)
        max_features = ['auto','sqrt','log2',None][np.random.randint(0, 3)]
        
        # fit the model on the training set with the parameters
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
        rf_model.fit(x_train, y_train)
        
        # make predictions on the test set
        y_pred = rf_model.predict(x_test)
        
        # assess the accuracy
        total_preds = 0
        total_correct = 0
        
        for j in range(0, y_pred.shape[0]):
            total_preds += 1
            if np.array_equal(y_pred[j], y_test.values[j]):
                total_correct += 1
    
        accuracy = (total_correct / total_preds)
        
        # determine whether to update parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            
            best_n_estimators = n_estimators
            best_max_depth = max_depth
            best_min_samples_split = min_samples_split
            best_min_samples_leaf = min_samples_leaf
            best_max_features = max_features
            
            # create a dictionary with the results
            best_params = {'n_estimators':best_n_estimators,
                   'max_depth':best_max_depth,
                   'min_samples_split':best_min_samples_split,
                   'min_samples_leaf':best_min_samples_leaf,
                   'max_features':best_max_features}

        logging.info('Completed hp tuning interation {}, best accuracy {} with params {}'.format(str(i+1), str(best_accuracy), best_params))
            
    
    
    # write parameters to disk
    output = json.dumps(best_params)
    f = open('best_params.json','w')
    f.write(output)
    f.close()
    
    # upload to cloud storage
    storage_client = storage.Client(project=project_id) 
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob('best_params.json')
    blob.upload_from_filename('best_params.json')
    
    return best_params