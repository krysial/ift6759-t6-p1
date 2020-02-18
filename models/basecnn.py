import typing
import datetime
import tensorflow as tf


class BaseCNNModel(tf.keras.Model):
    @classmethod
    def create(
        cls,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
    ):
        self = cls()
        self.cnn_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 5, strides=1, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(64, 5, strides=1, activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(128, 3, strides=(1, 1), activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=len(target_time_offsets))
        ])

        return self

    def call(self, inputs):
        return self.cnn_model(inputs['images'])
