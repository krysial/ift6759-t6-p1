import typing
import datetime
import tensorflow as tf


class BaseModel(tf.keras.Model):
    @classmethod
    def create(
        cls,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
    ):
        raise NotImplementedError("Should have implemented this")

    def save_weights(self, filepath, overwrite=True, save_format=None):
        # Because ModelCheckpointer set_model method overrides the save_weights_only flag
        super().save(
            filepath=filepath,
            overwrite=overwrite,
            include_optimizer=True,
            save_format='tf',
        )

