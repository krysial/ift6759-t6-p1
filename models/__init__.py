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
}


def prepare_model_eval(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    model_path = os.path.join(config['checkpoint_path'], config['model_id'] + ".tf/")
    assert os.path.exists(model_path), f"No model found in path:{model_path}"
    model = tf.keras.models.load_model(model_path, custom_objects={'scaled_rmse': scaled_rmse})

    return model


def prepare_model(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    if 'checkpoint_start' in config:
        model = tf.keras.models.load_model(
            os.path.join(
                config['checkpoint_path'],
                config['model_id'] + ".h5"
            )
        )
    else:
        model = models[config['model']].create(
            stations,
            target_time_offsets,
            config
        )

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
            num = tf.math.reduce_sum(tf.math.multiply((y_true - tf.reduce_mean(y_true, axis=0)), (y_pred - tf.reduce_mean(y_pred, axis=0))))
            den = tf.math.reduce_sum(tf.math.multiply(tf.math.square(y_true - tf.reduce_mean(y_true, axis=0)), tf.math.square(y_pred - tf.reduce_mean(y_pred, axis=0))))
            return tf.math.reduce_mean(num / den)

        optimizer = Adadelta(lr=config['learning_rate'])
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=[scaled_rmse],
        )

    return model
