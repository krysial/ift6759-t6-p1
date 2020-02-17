import typing
import datetime
import tensorflow as tf
from models.dummy import DummyModel
from models.lrcn import LRCNModel
from models.conv3d import Conv3DModel

models = {
    "dummy": DummyModel,
    "lrcn": LRCNModel,
    "conv3d": Conv3DModel
}


def prepare_model(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    return models[config.model].create(
        stations,
        target_time_offsets,
        config
    )
