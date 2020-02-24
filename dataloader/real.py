import pandas as pd
import typing
import datetime
import numpy as np

import itertools

from dataloader.get_GHI_targets import get_GHI_targets
from dataloader.get_raw_images import get_raw_images
from dataloader.get_raw_images import get_preprocessed_images
from dataloader.get_column_from_dataframe import get_column_from_dataframe


try:
    import pydevd
    DEBUGGING = True
except ImportError:
    DEBUGGING = False


def create_data_generator(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        station: typing.Dict[
            typing.AnyStr,
            typing.Tuple[float, float, float]
        ],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],):
    """
    A function to create a generator to yield data to the dataloader
    """
    def create_generator():
        if DEBUGGING:
            pydevd.settrace(suspend=False)

        # Order important to minimize file opening
        zip_target = zip(
            [final_dt for dt in target_datetimes for final_dt in (dt,) * len(station)],
            itertools.cycle(list(range(len(station))))  # Cycle through stations for each timedate
        )
        zip_len = len(target_datetimes) * len(station)

        for i in range(0, zip_len, config['batch_size']):
            datetimes_batch = list(itertools.islice(zip_target, i, i + config['batch_size']))

            images = get_preprocessed_images(
                dataframe,
                datetimes_batch,
                config
            )
            clearsky = get_column_from_dataframe(
                dataframe,
                datetimes_batch,
                station,
                target_time_offsets,
                'CLEARSKY_GHI',
                config
            )
            targets = get_GHI_targets(
                dataframe,
                datetimes_batch,
                station,
                target_time_offsets,
                config
            )

            yield {
                'images': images,
                'clearsky': clearsky,
            }, targets

    return create_generator
