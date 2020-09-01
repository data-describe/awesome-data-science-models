#!/usr/bin/env python
# coding: utf-8


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse
from google.cloud import storage
import glob
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.lib.io import file_io
import cv2
from sklearn.utils import class_weight, shuffle
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import urllib

WORKERS = 4
CHANNEL = 3
SIZE = 224
NUM_CLASSES = 1108
MODEL_CELLULAR ='CELLULAR.hdf5'
epochs = 2; batch_size = 8


class My_Generator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels,
                 batch_size, is_train=True,
                 mix=False, augment=False):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        if(self.is_train):
            self.on_epoch_end()
        self.is_mix = mix

    def __len__(self):
        try:
            
            return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))
        except:
            print('length error')

    def __getitem__(self, idx):
        try:
            batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

            if(self.is_train):
                return self.train_generate(batch_x, batch_y)
            return self.valid_generate(batch_x, batch_y)
        except:
            print('bad error')

    def on_epoch_end(self):
        if(self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
        else:
            pass
    
    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)        
        
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        
        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        try:
            client = storage.Client()
            batch_images = []
            for (sample, label) in zip(batch_x, batch_y):
                file = 'data/NWDATA/train/'+sample                
                source_bucket = client.get_bucket(args.bucket_name)
                source_blob = source_bucket.get_blob(file)
                image = np.asarray(bytearray(source_blob.download_as_string()), dtype="uint8")
                img = cv2.imdecode(image, cv2.IMREAD_COLOR)
                #print(img)
                #img = cv2.imread(bucket + '/data/NWDATA/train/'+sample)
                if(self.is_augment):
                    img = seq.augment_image(img)
                batch_images.append(img)
            batch_images = np.array(batch_images, np.float32)/255
            batch_y = np.array(batch_y, np.float32)
            if(self.is_mix):
                batch_images, batch_y = self.mix_up(batch_images, batch_y)
                print(len(batch_images))
            return batch_images, batch_y
        except:
            print('i am getting error')

    def valid_generate(self, batch_x, batch_y):
        try:
            client = storage.Client()
            batch_images = []
            for (sample, label) in zip(batch_x, batch_y):
                file = 'data/NWDATA/train/'+sample                
                source_bucket = client.get_bucket(args.bucket_name)
                source_blob = source_bucket.get_blob(file)
                image = np.asarray(bytearray(source_blob.download_as_string()), dtype="uint8")
                img = cv2.imdecode(image, cv2.IMREAD_COLOR)
                #print(img)
                #img = cv2.imread(bucket+'/data/NWDATA/train/'+sample)
                batch_images.append(img)
            batch_images = np.array(batch_images, np.float32)/255

            batch_y = np.array(batch_y, np.float32)
            return batch_images, batch_y
        except:
            print('valid error')

def create_model(input_shape,n_out):
    input_tensor = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.DenseNet121(include_top=False,
                       weights='imagenet',
                       input_tensor=input_tensor)
    #base_model.load_weights(bucket+'/DenseNet.h5')
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    final_output = tf.keras.layers.Dense(n_out, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(input_tensor, final_output)
    
    return model

def get_sirna(experiment):
    return experiment.split('_')[1]

def kappa_loss(y_true, y_pred, y_pow=2, eps=1e-12, N=1108, bsize=8, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
    
        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
    
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))
    
        return nom*0.5 / (denom + eps) + categorical_crossentropy(y_true, y_pred)*0.5

# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
        os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())

def wait_for_tpu_cluster_resolver_ready():

    """Waits for `TPUClusterResolver` to be ready and return it.
      Returns:
        A TPUClusterResolver if there is TPU machine (in TPU_CONFIG). Otherwise,
        return None.
      Raises:
        RuntimeError: if failed to schedule TPU.
      """
    
    tpu_config_env = os.environ.get('TPU_CONFIG')

    if not tpu_config_env:
        tf.logging.info('Missing TPU_CONFIG, use CPU/GPU for training.')
        return None

    tpu_node = json.loads(tpu_config_env)
    tf.logging.info('Waiting for TPU to be ready: \n%s.', tpu_node)

    num_retries = 40
    for i in range(num_retries):
        try:
            tpu_cluster_resolver = (
              tf.contrib.cluster_resolver.TPUClusterResolver(
                  tpu=[tpu_node['tpu_node_name']],
                  zone=tpu_node['zone'],
                  project=tpu_node['project'],
                  job_name='worker'))
            tpu_cluster_resolver_dict = tpu_cluster_resolver.cluster_spec().as_dict()
            if 'worker' in tpu_cluster_resolver_dict:
                tf.logging.info('Found TPU worker: %s', tpu_cluster_resolver_dict)
                return tpu_cluster_resolver
        except Exception as e:
            if i < num_retries - 1:
                tf.logging.info('Still waiting for provisioning of TPU VM instance.')
            else:
                # Preserves the traceback.
                raise RuntimeError('Failed to schedule TPU: {}'.format(e))
            time.sleep(10)

            # Raise error when failed to get TPUClusterResolver after retry.
            raise RuntimeError('Failed to schedule TPU.')
    
def train_and_evaluate(args):
    
    bucket = 'gs://' + str(args.bucket_name)
    
    df_train = pd.read_csv(bucket+'/data/new_train.csv')   
    df_train['sirna'] = df_train['sirna'].apply(get_sirna).apply(pd.to_numeric) 
    a = df_train['Filename'].str.contains('U2OS')
    b = df_train['Filename'].str.contains('HEPG2-01')


    df_new_train = df_train[a]
    df_new_train.reset_index(drop = True, inplace = True)
    df_new_train.shape
    df_train =df_new_train.copy()
    x = df_train['Filename']
    y = df_train['sirna']

    x, y = shuffle(x, y, random_state=10)
    #y.hist()

    y = pd.get_dummies(y)
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.10,
                                                        random_state=42)
    print(train_x.shape)
    print(train_y.shape)
    print(valid_x.shape)
    print(valid_y.shape)
    
    try:
        os.makedirs(args.job_dir)
    except:
        pass
    
    checkpoint_path = os.path.join(args.job_dir, MODEL_CELLULAR)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, 
                                save_best_only=True, mode='min', save_weights_only = True)

    reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                    verbose=1, mode='auto')

    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                        mode="min", 
                        patience=15)

    #csv_logger =  tf.keras.callbacks.CSVLogger(filename= bucket+'/data/training_log.csv',
    #                    separator=',',
    #                    append=True)

    train_generator = My_Generator(train_x, train_y, batch_size, is_train=True)
    train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False)
    valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)

    model = create_model(input_shape=(SIZE,SIZE,3),n_out=NUM_CLASSES)

    # warm up model
    for layer in model.layers:
        layer.trainable = False

    for i in range(-3,0):
        model.layers[i].trainable = True

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Nadam(1e-3))

    model.fit_generator(
        train_generator,
        steps_per_epoch=4,
        epochs=2,
        workers=WORKERS, use_multiprocessing=False,verbose=1)


    # train all layers
    for layer in model.layers:
        layer.trainable = True

    callbacks_list = [reduceLROnPlat]
    model.compile(optimizer=tf.keras.optimizers.Nadam(1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.fit_generator(
        train_mixup,
        #steps_per_epoch=np.ceil(float(len(train_y)) / float(256)),
        steps_per_epoch =4,
        validation_data=valid_generator,
        validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),
        epochs=epochs,
        verbose=1,
        workers=WORKERS, use_multiprocessing=False,
        callbacks=callbacks_list)


    if args.job_dir.startswith('gs://'):
        model.save(MODEL_CELLULAR)
        copy_file_to_gcs(args.job_dir, MODEL_CELLULAR)
    else:
        model.save(os.path.join(args.job_dir, MODEL_CELLULAR))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket_name',
        type=str,
        default='cellular-model-bucket',
        help='The Cloud Storage bucket to be used for process artifacts')

    parser.add_argument(
          '--job-dir',
          type=str,
          help='GCS or local dir to write checkpoints and export model',
          default='gs://sanofi-ml-workshop-chicago-taxi-demo/keras-job-dir')
    
    parser.add_argument(
          '--tputrain',
          type=str2bool, 
          nargs='?',        
          const=True, 
          default=False,)

    args, _ = parser.parse_known_args()
    
    print(args.tputrain)
    if args.tputrain ==True:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    train_and_evaluate(args)