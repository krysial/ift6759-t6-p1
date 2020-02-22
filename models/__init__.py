import typing
import datetime
import tensorflow as tf
from models.dummy import DummyModel
from models.lrcn import LRCNModel
from models.basecnn import BaseCNNModel
from models.se_res_bilrcn import SE_Residual_BiLRCNModel
from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.metrics import *

models = {
    "dummy": DummyModel,
    "lrcn": LRCNModel,
    "basecnn": BaseCNNModel,
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
