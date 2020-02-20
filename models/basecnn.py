import typing
import datetime
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *


class BaseCNNModel(tf.keras.Model):
    @classmethod
    def create(
        cls,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
    ):
        self = cls()
        self.cnn_model = Sequential([
            Conv2D(64, 5, strides=1, activation="relu"),
            Conv2D(64, 5, strides=1, activation="relu"),
            MaxPool2D(pool_size=2, strides=2),
            Conv2D(128, 3, strides=(1, 1), activation="relu"),
            MaxPool2D(pool_size=2, strides=2),
            Flatten(),
            Dense(units=512, activation="relu"),
            Dense(units=len(target_time_offsets))
        ])

        return self

    def call(self, inputs):
        return self.cnn_model(inputs['images'])
