import typing
import datetime

import pandas as pd
import tensorflow as tf

from dataloader.synthetic import create_synthetic_generator, Options


def prepare_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
) -> tf.data.Dataset:


        opts = Options(
            image_size=config.crop_size,
            digit_size=28,
            num_channels=len(config.channels),
            seq_len=config.seq_len,
            step_size=0.3,
            lat=station[config.station][0],
            lon=station[config.station][1],
            alt=station[config.station][2],
            offsets=target_time_offsets
        )
        generator = create_synthetic_generator(opts)
        data_loader = tf.data.Dataset.from_generator(
            generator, (tf.float32, tf.float32)
        ).batch(config.batch_size)