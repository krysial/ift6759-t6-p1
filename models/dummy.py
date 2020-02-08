import typing
import datetime
from models.base import BaseModel
import tensorflow as tf


class DummyModel(BaseModel):
    @classmethod
    def create(
        cls,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
    ):
        self = cls()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(len(target_time_offsets), activation=tf.nn.softmax)

        return self

    def call(self, inputs):
        x = self.dense1(self.flatten(inputs))
        return self.dense2(x)
