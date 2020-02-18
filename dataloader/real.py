import pandas as pd
import typing
import datetime
import numpy as np

from dataloader.get_GHI_targets import get_GHI_targets
from dataloader.get_raw_images import get_raw_images


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
            images = get_raw_images(dataframe, [datetime], config)
            yield images[0], targets[0]

    return create_generator
