
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# preprocessing
# =============
def create_scaler(data):
    '''Create standard scaler based on training data.'''
    data['native-country'] = data['native-country'].str.strip()
    # create scaler
    scale_cols = [
        'age', 'fnlwgt', 'education-num', 'capital-gain',
        'capital-loss', 'hours-per-week'
    ]
    scaler = StandardScaler()
    scaler.fit(data[scale_cols])
    return scaler

def preprocess(data, scaler):
    # remove string spaces
    for idx in data.dtypes.loc[data.dtypes == 'object'].index:
        data[idx] = data[idx].str.strip('. ')
    # make target numerical
    data['income_50k'] = (data.income == '>50K').astype('int')
    # make categoricals
    data['c_gain_ind'] = (data['capital-gain'] == 0).astype('int')
    data['c_loss_ind'] = (data['capital-loss'] == 0).astype('int')
    work_bins = list(np.linspace(0, 50, 6))+[np.inf]
    data['work_hrs_bins'] = pd.cut(data['hours-per-week'], bins=work_bins)
    # make dummy vars
    data = pd.get_dummies(
        data,
        columns=[
            'education', 'occupation', 'race', 'marital-status',
            'sex', 'relationship', 'workclass', 'native-country'
        ]
    )
    # standardize columns
    scale_cols = [
        'age', 'fnlwgt', 'education-num', 'capital-gain',
        'capital-loss', 'hours-per-week'
    ]
    data[scale_cols] = scaler.transform(data[scale_cols])
    # organize features
    ex_cols = ['income', 'native-country', 'income_50k', 'work_hrs_bins']
    data = data[['income_50k'] + list(data.columns.difference(ex_cols))]
    return data

def tf_dset_prep(dframe, shuffle_sz=10000, shuffle=True, batch_sz=32):
    '''Prepare tensorflow.data Dataset from prepped pandas dataframe'''
    dset = tf.data.Dataset.from_tensor_slices(dframe.values)
    dset = dset.map(lambda rec: (rec[1:], rec[0]))
    if shuffle==True:
        dset = dset.shuffle(shuffle_sz)
    dset = dset.batch(batch_sz).prefetch(1)
    return dset

# model build
def build_nn_model(input_sz, label, args):
    model = Sequential(name=label)
    sz = input_sz
    for _ in range(args.hidden_depth):
        layer = Dense(
            args.hidden_nodes, activation=args.hidden_activation,
            input_shape=(sz,), kernel_regularizer=args.kernel_regularizer
        )
        model.add(layer)
        sz = args.hidden_nodes
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(
        optimizer=args.optimizer,
        loss='binary_crossentropy',
        metrics=['AUC']  #tf.keras.metrics.AUC(curve='ROC')
    )
    return model

# model training
# ==============
def fit_model(model, train_data, validation_data, args):
    # callbacks
    stopper = EarlyStopping(
        monitor=args.metric, mode='max', patience=8, restore_best_weights=True
    )
    lr_scheduler = LearningRateScheduler(
        lambda epoch: args.initial_lr*(1 if epoch<10 else 10**((9-epoch)/args.lr_decay_param))
    )
    # custom callback used in hypertuning
    class HtuneCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            tf.summary.scalar('htune_ROC_AUC', logs.get(args.metric, 0.0), epoch)
    htuner_cb = HtuneCallback()
    logdir = "logs/scalars/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    weight1 = np.round(args.class_weight_ratio, 2)*10**2
    weights = {0:10**2, 1:weight1}

    # model training
    model.fit(
        train_data,
        class_weight=weights,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=[lr_scheduler, stopper, htuner_cb]
    )
    return model

# def save_model(model, model_uri):
#     model.save(model_uri, save_format='tf')