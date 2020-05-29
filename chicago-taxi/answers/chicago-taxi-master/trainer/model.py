from builtins import range

import keras

from keras import backend as K
from keras import layers
from keras import models

import logging
import numpy as np
import pandas as pd
import pandas_gbq
import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

# CSV columns in the input file.
CSV_COLUMNS = ('log_trip_seconds','distance','hour_start','month_start','weekday','pickup_census_tract','dropoff_census_tract','pickup_community_area','dropoff_community_area','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude')

LABEL_COLUMN = 'log_trip_seconds'

CAT_COLUMNS = ['weekday','pickup_census_tract','dropoff_census_tract','pickup_community_area','dropoff_community_area']

# generate a dictionary with unique values for each of the categorical features
cat_values = {}
for cat in CAT_COLUMNS:
    SQL = 'SELECT DISTINCT {} AS cat_feature FROM `mwpmltr.chicago_taxi.train_results`'.format(cat)
    df = pandas_gbq.read_gbq(SQL, project_id='mwpmltr')
    all_categorical_training = sorted(df['cat_feature'].tolist())
    cat_values[cat] = all_categorical_training


def returnUniqueCount(feature):
  """Returns the ount of distinct values for a given feature"""
  SQL = 'SELECT COUNT(DISTINCT {}) AS unique_count FROM `mwpmltr.chicago_taxi.train_results`'.format(feature)
  df = pandas_gbq.read_gbq(SQL, project_id='mwpmltr')
    
  return df['unique_count'][0]


def model_fn(learning_rate,
             num_deep_layers,
             first_deep_layer_size,
             first_wide_layer_size,
             wide_scale_factor,
             dropout_rate):
  """Create a Keras Wide and Deep model with layers and the Functional API.

  Args:
    learning_rate: [float] the learning rate for the optimizer.
    num_deep_layers: [int] the number of layers in the deep neural network
    first_deep_layer_size: [int] the number of nodes in the first hidden layer of the deep neural network
    first_wide_layer_size: [int] the number of nodes in the first hidden layer of the wide neural network
    wide_scale_factor: [float] to scale factor used to decrease the number of nodes in each successive layer of the wide neural network

  Returns:
    A Keras model.
  """

  num_continuous_features = len(CSV_COLUMNS) - len(CAT_COLUMNS) - 1 # label

  # "set_learning_phase" to False to avoid:
  # AbortionError(code=StatusCode.INVALID_ARGUMENT during online prediction.
  K.set_learning_phase(False)
  # define the wide model
  weekday_count = returnUniqueCount('weekday')
  weekday_inputs = layers.Input(shape=(weekday_count,))

  pickup_census_tract_count = returnUniqueCount('pickup_census_tract')
  pickup_census_tract_inputs = layers.Input(shape=(pickup_census_tract_count,))

  dropoff_census_tract_count = returnUniqueCount('dropoff_census_tract')
  dropoff_census_tract_inputs = layers.Input(shape=(dropoff_census_tract_count,))

  pickup_community_area_count = returnUniqueCount('pickup_community_area')
  pickup_community_area_inputs = layers.Input(shape=(pickup_community_area_count,))

  dropoff_community_area_count = returnUniqueCount('dropoff_community_area')
  dropoff_community_area_inputs = layers.Input(shape=(dropoff_community_area_count,))

  merged_layer = layers.concatenate([weekday_inputs, pickup_census_tract_inputs, dropoff_census_tract_inputs, pickup_community_area_inputs, dropoff_community_area_inputs])
  merged_layer = layers.Dense(first_wide_layer_size, activation='relu')(merged_layer)
  merged_layer = layers.Dropout(dropout_rate)(merged_layer)

    # adjust architecture of the wide model based on input features
  i = 1
  num_nodes = max(num_continuous_features, int(first_wide_layer_size * wide_scale_factor**i))
  output = layers.Dense(num_nodes, activation='relu')(merged_layer)
  output = layers.Dropout(dropout_rate)(output)
  while num_nodes > num_continuous_features:
    i+=1
    num_nodes = max(num_continuous_features, int(first_wide_layer_size * wide_scale_factor**i))
    output = layers.Dense(num_nodes, activation='relu')(output)
    output = layers.Dropout(dropout_rate)(output)

  wide_model = models.Model(inputs=[weekday_inputs, pickup_census_tract_inputs, dropoff_census_tract_inputs, pickup_community_area_inputs, dropoff_community_area_inputs], outputs=output)
  wide_model = compile_model(wide_model, learning_rate)
  #logging.info(wide_model.summary())


  # define the deep model
  deep_inputs = layers.Input(shape=(num_continuous_features,)) # 7 continous features in current model
  #output = layers.Dense(num_continuous_features, activation='relu')(deep_inputs)
  deep_model = models.Model(inputs=deep_inputs, outputs=deep_inputs) 
  deep_model = compile_model(deep_model, learning_rate)
  #logging.info(deep_model.summary())

  
  # combine the two models
  # Combine wide and deep into one model
  merged_out = layers.concatenate([wide_model.output, deep_model.output])
    # adjust architecture of the deep model based on input features
  for i in range(1, num_deep_layers + 1):
    num_nodes = max(2, int(first_deep_layer_size / (i)))
    if i == 1:
      output = layers.Dense(num_nodes, activation='relu')(merged_out)
    else:
      output = layers.Dense(num_nodes, activation='relu')(output)
    output = layers.Dropout(dropout_rate)(output)

  predictions = layers.Dense(1, activation='linear')(output)
  combined_model = models.Model([deep_model.input] + wide_model.input, predictions)

  combined_model = compile_model(combined_model, learning_rate)
  logging.info(combined_model.summary())


  return combined_model


def compile_model(model, learning_rate):
  model.compile(
      loss='mse',
      optimizer=keras.optimizers.Adam(lr=learning_rate), # , clipnorm=1. 
      metrics=['mae', 'mse'])
  return model


def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(
      inputs={'input': model.inputs[0]}, outputs={'mpg_pred': model.outputs[0]})

  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        })
    builder.save()


def to_categorical_features(features, categorical_feature):
  """Converts categorical features into a one-hot encoding representation

  Args:
    features: Input features in the data 
    categorical_feature: Categorical feature to be one-hot encoded

  Returns:
    features: The original input features minus the categorical_feature
    categorical_dummies: A one-hot encoded representation of the categorical feature
  """

  # collect unique categorical features
  all_categorical_training = cat_values[categorical_feature]

  categorical_dummies = pd.get_dummies(features[categorical_feature], drop_first=False)

  # add in any missing categorical columns
  for cat in all_categorical_training:
    if cat not in categorical_dummies.columns:
      categorical_dummies[cat] = 0

  # remove any categorical features that aren't included in the training set
  for cat in categorical_dummies.columns:
    if cat not in all_categorical_training:
      categorical_dummies = categorical_dummies.drop([cat], axis=1)

  # reorder the columns
  categorical_dummies = categorical_dummies[all_categorical_training]

  # drop the original column from the features
  features = features.drop([categorical_feature], axis=1)
  # convert feature values to floats
  features = features.astype(float)

  # return the updated original feature set along with the categorical features
  return features, categorical_dummies


def to_numeric_features(features):
  """Converts the pandas input features to continous and categorical values.

  Args:
    features: Input features in the data 

  Returns:
    A pandas dataframe for each of the categorical variables and one for the continous variables
  """

  # begin extracting categorical features
  features, weekday_cat = to_categorical_features(features, 'weekday')
  features, pickup_census_tract_cat = to_categorical_features(features, 'pickup_census_tract')
  features, dropoff_census_tract_cat = to_categorical_features(features, 'dropoff_census_tract')
  features, pickup_community_area_cat = to_categorical_features(features, 'pickup_community_area')
  features, dropoff_community_area_cat = to_categorical_features(features, 'dropoff_community_area')

  # replace NaN values with zeros in the continous features
  features = features.fillna(0)

  return features, weekday_cat, pickup_census_tract_cat, dropoff_census_tract_cat, pickup_community_area_cat, dropoff_community_area_cat
  


def generator_input(filenames, chunk_size, project_id, bucket_name, x_scaler, batch_size=64):
  """Produce features and labels needed by keras fit_generator."""

  while True:
    input_reader = pd.read_csv(
        tf.io.gfile.GFile(filenames[0]),
        names=CSV_COLUMNS,
        chunksize=chunk_size)
        #,na_values=0)

    for input_data in input_reader:
      # clean up any rows that contain column headers (in case they were inserted during the sharding and recombining)
      input_data['log_trip_seconds_num'] = input_data['log_trip_seconds'].apply(lambda x: x!='log_trip_seconds')

      input_data = input_data[input_data['log_trip_seconds_num'] == True]
      input_data = input_data.drop(['log_trip_seconds_num'],axis=1)

      # separate label from feature columns
      label = input_data.pop(LABEL_COLUMN).astype(float)
      features, weekday_cat, pickup_census_tract_cat, dropoff_census_tract_cat, pickup_community_area_cat, dropoff_community_area_cat = to_numeric_features(input_data)

      features_scaled = x_scaler.transform(features)
      features = pd.DataFrame(features_scaled, columns=list(features.columns))

      # generate output in batches of length batch_size
      idx_len = input_data.shape[0]
      for index in range(0, idx_len, batch_size):
        features_part = features.iloc[index:min(idx_len, index + batch_size)]
        features_part = np.nan_to_num(features_part.values)
        
        label_part = label.iloc[index:min(idx_len, index + batch_size)]
        label_part = label_part.astype(float)

        weekday_cat_part = weekday_cat.iloc[index:min(idx_len, index + batch_size)]
        weekday_cat_part = np.nan_to_num(weekday_cat_part.values)

        pickup_census_tract_cat_part = pickup_census_tract_cat.iloc[index:min(idx_len, index + batch_size)]
        pickup_census_tract_cat_part = np.nan_to_num(pickup_census_tract_cat_part.values)

        dropoff_census_tract_cat_part = dropoff_census_tract_cat.iloc[index:min(idx_len, index + batch_size)]
        dropoff_census_tract_cat_part = np.nan_to_num(dropoff_census_tract_cat_part.values)

        pickup_community_area_cat_part = pickup_community_area_cat.iloc[index:min(idx_len, index + batch_size)]
        pickup_community_area_cat_part = np.nan_to_num(pickup_community_area_cat_part.values)

        dropoff_community_area_cat_part = dropoff_community_area_cat.iloc[index:min(idx_len, index + batch_size)]
        dropoff_community_area_cat_part = np.nan_to_num(dropoff_community_area_cat_part.values)


        x = [features_part, weekday_cat_part, pickup_census_tract_cat_part, dropoff_census_tract_cat_part, pickup_community_area_cat_part, dropoff_community_area_cat_part] 
        y = label_part

        yield (x, y)  
