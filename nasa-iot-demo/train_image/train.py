# import libraries
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from google.cloud import storage
import fire
import subprocess
import pickle
import time


def train(bucket, train_file, test_file, job_dir, epochs=100, batch_size=10):
    # Read in the data
    train_loc = "gs://{0}/{1}".format(bucket, train_file)
    train_data = pd.read_csv(train_loc, index_col=0)
    X_train = train_data.to_numpy()
    
    test_loc = "gs://{0}/{1}".format(bucket, test_file)
    test_data = pd.read_csv(test_loc, index_col=0)
    X_test = test_data.to_numpy()
    
    # reshape inputs for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    print("Training data shape:", X_train.shape)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print("Test data shape:", X_test.shape)   
    
    
    # define the model network for the autoencoder
    def autoencoder_model(X):
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True, 
                  kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(X.shape[2]))(L5)    
        model = Model(inputs=inputs, outputs=output)
        return model
    
    
    # create the autoencoder model
    model = autoencoder_model(X_train)
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    
    # fit the model to the data
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1).history
    
    
    def save_to_gcs(job_dir, folder, filename):
        gcs_path = "{0}/{1}/{2}".format(job_dir, folder, filename)
        subprocess.check_call(['gsutil', 'cp', filename, gcs_path], stderr=sys.stdout)
        
    
    # Save the model to the job dir and GCS
    model_filename = "model.h5"
    model.save(model_filename)
    save_to_gcs(job_dir, "model", model_filename)
    #gcs_model_path = "{0}/model/{1}".format(job_dir, model_filename)
    #subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
    
    # Save the metrics to the job dir and GCS
    columns = ['loss', 'val_loss']
    metrics = pd.DataFrame(columns=columns)
    metrics['loss'] = history['loss']
    metrics['val_loss'] = history['val_loss']
    metrics_filename = "metrics.csv"
    metrics.to_csv(metrics_filename, index=False)
    save_to_gcs(job_dir, "metrics", metrics_filename)
    #gcs_metrics_path = "{0}/metrics/{1}".format(job_dir, metrics_filename)
    #subprocess.check_call(['gsutil', 'cp', metrics_filename, gcs_metrics_path], stderr=sys.stdout)
    
    # Evaluate the model
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mae)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    loss_fig_filename = "loss.png"
    fig.savefig(loss_fig_filename)
    save_to_gcs(job_dir, "artifacts", loss_fig_filename)
    
    # plot the loss distribution of the training set
    X_pred = model.predict(X_train)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns=train_data.columns)
    X_pred.index = train_data.index

    scored = pd.DataFrame(index=train_data.index)
    Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
    plt.xlim([0.0,.5])
    loss_dist_filename = "loss_dist.png"
    fig.savefig(loss_dist_filename)
    save_to_gcs(job_dir, "artifacts", loss_dist_filename)
    
    # calculate the loss on the test set
    X_pred = model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns=test_data.columns)
    X_pred.index = test_data.index

    scored = pd.DataFrame(index=test_data.index)
    Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
    #scored['Threshold'] = 0.275
    #scored['Threshold'] = 1.05 * scored['Loss_mae'].max()  # Give a 5% buffer to the value
    #scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    
    # calculate the same metrics for the training set 
    # and merge all data in a single dataframe for plotting
    X_pred_train = model.predict(X_train)
    X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
    X_pred_train = pd.DataFrame(X_pred_train, columns=train_data.columns)
    X_pred_train.index = train_data.index
    
    scored_train = pd.DataFrame(index=train_data.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis = 1)
    scored = pd.concat([scored_train, scored])
    scored = scored.reset_index(drop=True)
    #scored['Threshold'] = 0.275
    scored['Threshold'] = 1.05 * scored_train['Loss_mae'].max()  # Give a 5% buffer to the value
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    
    # plot bearing failure time plot
    fail_plot_filename = "bearing_failure.png"
    ax = scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])
    fig = ax.get_figure()
    fig.savefig(fail_plot_filename)
    save_to_gcs(job_dir, "artifacts", fail_plot_filename)
    
    
if __name__ == "__main__":
    fire.Fire(train)
