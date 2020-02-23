import pandas as pd
import typing
import datetime
import numpy as np

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

        for i, datetime in enumerate(target_datetimes):
            targets = get_GHI_targets(
                dataframe,
                [datetime],
                station,
                target_time_offsets,
                config
            )
            images = get_preprocessed_images(
                dataframe,
                [datetime],
                config,
                station
            )
            clearsky = get_column_from_dataframe(
                dataframe,
                [datetime],
                station,
                target_time_offsets,
                'CLEARSKY_GHI',
                config
            )
            yield {
                'images': images[0],
                'clearsky': clearsky[0],
            }, targets[0]

    return create_generator
