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
        no_channels = len(config['channels'])
        seq_len = config['no_of_temporal_seq']
        # 40 40
        tf.keras.Sequential([
            tf.keras.layers.Conv2D(no_channels * seq_len, 11, strides=(4, 4), activation=tf.nn.relu, input_shape=(None, None, no_channels * seq_len)),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(no_channels * seq_len, 5, strides=(2, 2), activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(no_channels * seq_len, 3, strides=(1, 1), activation=tf.nn.relu),
          # tf.keras.layers.Conv2D(no_channels * seq_len, 3, strides=(1, 1), activation=tf.nn.relu)
          # tf.keras.layers.Conv2D(no_channels * seq_len, 3, strides=(1, 1), activation=tf.nn.relu)
          # tf.keras.layers.Conv2D(no_channels * seq_len, 3, strides=(1, 1), activation=tf.nn.relu),
          # tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            #tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
          # tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
            tf.keras.layers.Dense(units=len(target_time_offsets))
        ])

    def call(self, inputs):

        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.mean_squared_error()

        loss_history = []
        return self.dense3(x)
