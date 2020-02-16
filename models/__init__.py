import typing
import datetime
import tensorflow as tf
from models.dummy import DummyModel
from models.lrcn import LRCNModel

models = {
    "dummy": DummyModel,
    "lrcn": LRCNModel
}


def prepare_model(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    return models[config["model"]].create(
        stations,
        target_time_offsets,
        config
    )
