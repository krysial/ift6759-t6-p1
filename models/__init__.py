import typing
import datetime
import tensorflow as tf
from models.dummy import DummyModel
from models.lrcn import LRCNModel
from models.basecnn import BaseCNNModel
from models.conv3d import Conv3DModel
from tensorflow.keras.optimizers import *
import os

models = {
    "dummy": DummyModel,
    "lrcn": LRCNModel,
    "basecnn": BaseCNNModel,
    "cnn3d": Conv3DModel,
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
        optimizer = Adam(lr=config['learning_rate'], decay=config['decay_rate'])
        model = models[config['model']].create(
            stations,
            target_time_offsets,
            config
        )

        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer
        )

    return model
