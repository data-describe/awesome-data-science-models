import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.lib.io import file_io

from sklearn.model_selection import train_test_split

import argparse


BATCH_SIZE = 32
IMAGE_SIZE = [200, 200]
MODEL_MALARIA ='Malaria.hdf5'

def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def pad(image, label):
    image, label = convert(image, label)
    image = tf.image.resize_with_crop_or_pad(image, 200, 200)
    return image, label


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )

    return block


def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])

    return block


def build_model():

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),

        conv_block(32),
        conv_block(64),

        conv_block(128),
        tf.keras.layers.Dropout(rate=0.2),

        conv_block(256),
        tf.keras.layers.Dropout(rate=0.2),

        tf.keras.layers.Flatten(),
        dense_block(512, 0.2),
        dense_block(128, 0.2),
        dense_block(64, 0.2),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def train_and_evaluate(args):
    
    if args.tputrain ==True:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()

    
    train_ds, val_ds = tfds.load('malaria', split=['train[:5%]', 'train[10%:12%]'], shuffle_files=True,download =False,
                                 as_supervised=True,data_dir="gs://malaria_demo")
        
    padded_train_ds = (
        train_ds
            .cache()
            .map(pad)
            .batch(BATCH_SIZE)
    )

    padded_val_ds = (
        val_ds
            .cache()
            .map(pad)
            .batch(BATCH_SIZE)
    )
    
    try:
        os.makedirs(args.job_dir)
    except:
        pass
    
    checkpoint_path = os.path.join(args.job_dir, MODEL_MALARIA)
    
    
    if args.tputrain ==True:
        with strategy.scope():
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                       save_best_only=True)

            early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                         restore_best_weights=True)

            exponential_decay_fn = exponential_decay(0.01, 20)

            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    
            model = build_model()
            model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                       save_best_only=True)

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                         restore_best_weights=True)

        exponential_decay_fn = exponential_decay(0.01, 20)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
            
        model = build_model()
        model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        

    history = model.fit(padded_train_ds, epochs=5,
                        validation_data=padded_val_ds, callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler])
    
    if args.job_dir.startswith('gs://'):
        model.save(MODEL_MALARIA)
        copy_file_to_gcs(args.job_dir, MODEL_MALARIA)
    else:
        model.save(os.path.join(args.job_dir, MODEL_MALARIA))
        
    
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
        os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())
            

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
          default='gs://bipin_bucket/keras-job-dir')
    
    parser.add_argument(
          '--tputrain',
          type=str2bool, 
          nargs='?',        
          const=True, 
          default=False,)

    args, _ = parser.parse_known_args()
    
    print(args.tputrain)
    train_and_evaluate(args)


    