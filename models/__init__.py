import typing
import datetime
import tensorflow as tf
from models.dummy import DummyModel
from models.lrcn import LRCNModel
from models.basecnn import BaseCNNModel
from models.conv3d import Conv3DModel
from models.se_res_bilrcn import SE_Residual_BiLRCNModel
from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.metrics import *

models = {
    "dummy": DummyModel,
    "lrcn": LRCNModel,
    "basecnn": BaseCNNModel,
    "cnn3d": Conv3DModel,
    "se_res_bilrcn": SE_Residual_BiLRCNModel,
}


def prepare_model_eval(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    return tf.keras.models.load_model(
        os.path.join(
            config['checkpoint_path'],
            config['model_id'] + ".h5"
        )
    )


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
            metrics=['RootMeanSquaredError', rse, CORR],
        )

    return model
