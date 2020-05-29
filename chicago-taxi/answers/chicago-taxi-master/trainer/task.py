import argparse
from google.cloud import storage
import glob
import joblib
import logging
import os
from os import path

#from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

from tensorflow.python.lib.io import file_io


# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
CHUNK_SIZE = 5000
CHECKPOINT_FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
CENSUS_MODEL = 'census.hdf5'

def train_and_evaluate(args):

  # confirm whether training datasets need to be created
  if args.create_data == True:
    import trainer.create_data_func as create_data_func

    logging.info('Begin creating datasets')
    for data_part in ['train','val','test']:
      create_data_func.create_data_func(data_part, args.project_id, args.bucket_name, args.dataset_id)
      
    logging.info('End creating datasets')

  # import after datasets are created as they are referenced immediately when this module is initiated
  import trainer.model as model

  # if new datasets are created, scaler also need to be created
  if args.create_data == True:
    import trainer.create_scaler_func as create_scaler_func

    logging.info('Begin fitting scaler')
    create_scaler_func.create_scaler_func(args.train_files, model.CSV_COLUMNS, model.LABEL_COLUMN, args.bucket_name, args.project_id)
      
    logging.info('End fitting scalers')

  # download the scaler
  if not path.exists('x_scaler'):
    logging.info('Downloading scaler')
    storage_client = storage.Client(project=args.project_id)
    bucket = storage_client.get_bucket(args.bucket_name)
    blob = bucket.blob('scalers/x_scaler')
    blob.download_to_filename('x_scaler')
    logging.info('Downloaded scaler')

  x_scaler = joblib.load('x_scaler')

  # build the model 
  census_model = model.model_fn(learning_rate=args.learning_rate, num_deep_layers=args.num_deep_layers, first_deep_layer_size=args.first_deep_layer_size, first_wide_layer_size=args.first_wide_layer_size, wide_scale_factor=args.wide_scale_factor, dropout_rate=args.dropout_rate)
  logging.info(census_model.summary())

  try:
    os.makedirs(args.job_dir)
  except:
    pass
  
  checkpoint_path = os.path.join(args.job_dir, CHECKPOINT_FILE_PATH)

  # Model checkpoint callback.
  checkpoint = ModelCheckpoint(
      checkpoint_path,
      monitor='val_mse', # 'mean_squared_error'
      verbose=1,
      period=args.checkpoint_epochs,
      save_best_only=True,
      mode='min')


  # Early stopping callback.
  early_stop = EarlyStopping(monitor='val_mse', patience=10)  # 'mean_squared_error'

  # Tensorboard logs callback.
  tb_log = TensorBoard(
      log_dir=os.path.join(args.job_dir, 'logs'),
      histogram_freq=0,
      write_graph=True,
      embeddings_freq=0)

  callbacks = [checkpoint, early_stop, tb_log]

  # fit the model on the training set
  census_model.fit_generator(
      generator=model.generator_input(args.train_files, chunk_size=CHUNK_SIZE, project_id=args.project_id, bucket_name=args.bucket_name, x_scaler=x_scaler),
      steps_per_epoch=args.train_steps, 
      epochs=args.num_epochs,
      callbacks=callbacks,
      validation_data=model.generator_input(args.eval_files, chunk_size=CHUNK_SIZE, project_id=args.project_id, bucket_name=args.bucket_name, x_scaler=x_scaler),
      validation_steps=args.eval_steps)

  # evaluate model on test set
  loss, mae, mse  = census_model.evaluate_generator(
            model.generator_input(args.test_files, chunk_size=CHUNK_SIZE, project_id=args.project_id, bucket_name=args.bucket_name, x_scaler=x_scaler),
            steps=args.test_steps)
  logging.info('\nTest evaluation metrics[{:.2f}, {:.2f}, {:.2f}] {}'.format(loss, mae, mse, census_model.metrics_names))

  # Unhappy hack to workaround h5py not being able to write to GCS.
  # Force snapshots and saves to local filesystem, then copy them over to GCS.
  if args.job_dir.startswith('gs://'):
    census_model.save(CENSUS_MODEL)
    copy_file_to_gcs(args.job_dir, CENSUS_MODEL)
  else:
    census_model.save(os.path.join(args.job_dir, CENSUS_MODEL))

  # Convert the Keras model to TensorFlow SavedModel.
  model.to_savedmodel(census_model, os.path.join(args.job_dir, 'export'))


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
  with file_io.FileIO(file_path, mode='rb') as input_f:
    with file_io.FileIO(
        os.path.join(job_dir, file_path), mode='w+') as output_f:
      output_f.write(input_f.read())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train-files',
      nargs='+',
      help='Training file local or GCS',
      default=['gs://ross-keras/data/full_train_results.csv'])
  parser.add_argument(
      '--eval-files',
      nargs='+',
      help='Evaluation file local or GCS',
      default=['gs://ross-keras/data/full_val_results.csv'])
  parser.add_argument(
      '--test-files',
      nargs='+',
      help='Test file local or GCS',
      default=['gs://ross-keras/data/full_test_results.csv'])
  parser.add_argument(
      '--job-dir',
      type=str,
      help='GCS or local dir to write checkpoints and export model',
      default='gs://ross-keras/keras-job-dir')
  parser.add_argument(
      '--train-steps',
      type=int,
      default=20,
      help="""\
        Maximum number of training steps to perform
        Training steps are in the units of training-batch-size.
        So if train-steps is 500 and train-batch-size if 100 then
        at most 500 * 100 training instances will be used to train.""")
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=20,
      type=int)
  parser.add_argument(
      '--test-steps',
      help='Number of steps to run test for after model training',
      default=500,
      type=int)
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=50,
      help='Batch size for training steps')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=132,
      help='Batch size for evaluation steps')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=.0001,
      help='Learning rate for SGD')
  parser.add_argument(
      '--eval-frequency',
      default=10,
      help='Perform one evaluation per n epochs')
  parser.add_argument(
      '--first-deep-layer-size',
      type=int,
      default=256,
      help='Number of nodes in the first layer of DNN')
  parser.add_argument(
      '--num-deep-layers',
      type=int,
      default=3,
      help='Number of layers in DNN')
  parser.add_argument(
      '--first-wide-layer-size',
      type=int,
      default=256,
      help='Number of nodes in the first layer of Wide Neural Net')
  parser.add_argument(
      '--wide-scale-factor',
      type=float,
      default=0.25,
      help="""Rate of decay size of layer for Wide Neural Net.
        max(2, int(first_layer_size * scale_factor**i))""")
  parser.add_argument(
      '--eval-num-epochs',
      type=int,
      default=50,
      help='Number of epochs during evaluation')
  parser.add_argument(
      '--num-epochs',
      type=int,
      default=50,
      help='Maximum number of epochs on which to train')
  parser.add_argument(
      '--checkpoint-epochs',
      type=int,
      default=5,
      help='Checkpoint per n training epochs')
  parser.add_argument(
    '--create-data',
    type=bool,
    default=False,
    help='Whether or not to create data for train, test, and validation')
  parser.add_argument(
    '--dropout-rate',
    type=float,
    default=.2,
    help='Whether or not to create data for train, test, and validation')
  parser.add_argument(
    '--project-id',
    type=str,
    default='mwpmltr',
    help='The GCP Project ID')
  parser.add_argument(
    '--bucket-name',
    type=str,
    default='ross-keras',
    help='The Cloud Storage bucket to be used for process artifacts')
  parser.add_argument(
    '--dataset-id',
    type=str,
    default='chicago_taxi',
    help='The Dataset ID to be used in BigQuery for storing preprocessed data.')


  args, _ = parser.parse_known_args()
  train_and_evaluate(args)
