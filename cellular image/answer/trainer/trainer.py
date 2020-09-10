# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a simple model on the Malaria dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
from official.vision.image_classification import common

FLAGS = flags.FLAGS
train_length = 22046
test_length = 1378

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

def build_model():
    """Constructs the ML model used to predict handwritten digits."""

    image = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    y = tf.keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             activation='relu')(image)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
    y = tf.keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             activation='relu')(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(1024, activation='relu')(y)
    y = tf.keras.layers.Dropout(0.4)(y)

    probs = tf.keras.layers.Dense(1, activation='sigmoid')(y)

    model = tf.keras.models.Model(image, probs, name='malaria')

    return model


@tfds.decode.make_decoder(output_dtype=tf.float32)
def decode_image(example, feature):
    """Convert image to float32 and normalize from [0, 255] to [0.0, 1.0]."""
    return tf.cast(feature.decode_example(example), dtype=tf.float32) / 255


def run(flags_obj, datasets_override=None, strategy_override=None):
    """Run Malaria model training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.
    datasets_override: A pair of `tf.data.Dataset` objects to train the model,
                       representing the train and test sets.
    strategy_override: A `tf.distribute.Strategy` object to use for model.

  Returns:
    Dictionary of training and eval stats.
  """
    strategy = strategy_override or distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      tpu_address=flags_obj.tpu)

    strategy_scope = distribution_utils.get_strategy_scope(strategy)

    train_ds, val_ds = tfds.load('malaria', split=['train[:5%]', 'train[90%:92%]'], shuffle_files=True,download =False,
                                 as_supervised=True,data_dir=flags_obj.data_dir)      
  
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
    
    ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
    with strategy_scope:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.05, decay_steps=100000, decay_rate=0.96)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
                                                       save_best_only=True)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        model = build_model()
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

    train_steps = train_length // flags_obj.batch_size
    train_epochs = flags_obj.train_epochs

    #num_eval_examples = val_ds.info.splits['train'].num_examples
    num_eval_steps = test_length // flags_obj.batch_size
    
    callbacks = [checkpoint_cb]
    
    history = model.fit(
          padded_train_ds,
          epochs=3,
          callbacks=callbacks,
          validation_data=padded_val_ds)

    export_path = os.path.join(flags_obj.model_dir, 'saved_model')
    model.save(export_path, include_optimizer=False)

    eval_output = model.evaluate(
          padded_val_ds, steps=num_eval_steps, verbose=2)

    stats = common.build_stats(history, eval_output, callbacks)
    return stats


def define_malaria_flags():
    """Define command line flags for Malaria model."""
    flags_core.define_base(
          clean=True,
          num_gpu=True,
          train_epochs=True,
          epochs_between_evals=True,
          distribution_strategy=True)
    flags_core.define_device()
    flags_core.define_distribution()
    flags.DEFINE_bool('download', False,
                    'Whether to download data to `--data_dir`.')
    FLAGS.set_default('batch_size', 1024)


def main(_):
    model_helpers.apply_clean(FLAGS)
    stats = run(flags.FLAGS)
    logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_malaria_flags()
    app.run(main)
