import typing
import datetime
import tensorflow as tf
import numpy as np
from models.dummy import DummyModel
from models.lrcn import LRCNModel
from models.basecnn import BaseCNNModel
from models.convlstm import ConvLSTMModel
from models.conv3d import Conv3DModel
from models.se_res_bilrcn import SE_Residual_BiLRCNModel
from models.bilrcn import BiLRCNModel
from utils.rescale_GHI import rescale_GHI
from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *


models = {
    "dummy": DummyModel,
    "lrcn": LRCNModel,
    "basecnn": BaseCNNModel,
    "convlstm": ConvLSTMModel,
    "cnn3d": Conv3DModel,
    "se_res_bilrcn": SE_Residual_BiLRCNModel,
    "bilrcn": BiLRCNModel,
}


# metrics
def scaled_rmse(y_true, y_pred):
    y_true_ = (y_true * 470.059048) + 297.487143
    y_pred_ = (y_pred * 470.059048) + 297.487143

    mse = tf.math.reduce_mean(tf.math.square(y_true_ - y_pred_))
    rmse_loss = tf.math.sqrt(mse)
    return rmse_loss


# Root Relative Squared Error
def rse(y_true, y_pred):
    num = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_true - y_pred)))
    den = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - tf.math.reduce_mean(y_true, axis=None))))
    return tf.math.reduce_mean(num / den)


# Empirical Correlation Coefficient
def CORR(y_true, y_pred):
    num = tf.math.reduce_sum(
        tf.math.multiply((y_true - tf.reduce_mean(y_true, axis=0)), (y_pred - tf.reduce_mean(y_pred, axis=0))))
    den = tf.math.reduce_sum(tf.math.multiply(tf.math.square(y_true - tf.reduce_mean(y_true, axis=0)),
                                              tf.math.square(y_pred - tf.reduce_mean(y_pred, axis=0))))
    return tf.math.reduce_mean(num / den)


def prepare_model_eval(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    model = models[config['model']].create(
        stations,
        target_time_offsets,
        config
    )

    images = np.zeros(
        (1,
            config['seq_len'],
            config['crop_size'],
            config['crop_size'],
            len(config['channels']))
    )

    clearsky = np.zeros(
        (1, len(target_time_offsets))
    )

    model({
        'images': images,
        'clearsky': clearsky
    })

    model.load_weights(
        os.path.join(config['checkpoint_path'], config['model_id'] + ".hdf5")
    )

    return model


def prepare_model(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    model = models[config['model']].create(
        stations,
        target_time_offsets,
        config
    )

    images = np.zeros(
        (1,
            config['seq_len'],
            config['crop_size'],
            config['crop_size'],
            len(config['channels']))
    )
    clearsky = np.zeros(
        (1, len(target_time_offsets))
    )

    model({
        'images': images,
        'clearsky': clearsky
    })

    if 'checkpoint_start' in config:
        model.load_weights(
            os.path.join(config['checkpoint_path'], config['model_id'] + ".h5")
        )

    if config['optimizer'] == 'Adadelta':
        optimizer = Adadelta(lr=config['learning_rate'])
    if config['optimizer'] == 'Adam':
        optimizer = Adam(lr=config['learning_rate'], decay=config['decay_rate'])

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=[scaled_rmse],
    )

    return model
