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

        def batch_datetimes():
            filtered_df = []
            for index, row in dataframe.iterrows():
                if row['BND_DAYTIME'] == 1:
                    filtered_df.append((index, 0))
                if row['TBL_DAYTIME'] == 1:
                    filtered_df.append((index, 1))
                if row['DRA_DAYTIME'] == 1:
                    filtered_df.append((index, 2))
                if row['FPK_DAYTIME'] == 1:
                    filtered_df.append((index, 3))
                if row['GWN_DAYTIME'] == 1:
                    filtered_df.append((index, 4))
                if row['PSU_DAYTIME'] == 1:
                    filtered_df.append((index, 5))
                if row['SXF_DAYTIME'] == 1:
                    filtered_df.append((index, 6))

                if len(filtered_df) > config['batch_size']:
                    batch = filtered_df[:config['batch_size']]
                    filtered_df[config['batch_size']:]
                    yield batch

        for batch in batch_datetimes():
            images = get_preprocessed_images(
                dataframe,
                batch,
                config
            )
            clearsky = get_column_from_dataframe(
                dataframe,
                batch,
                station,
                target_time_offsets,
                'CLEARSKY_GHI',
                config
            )
            targets = get_GHI_targets(
                dataframe,
                batch,
                station,
                target_time_offsets,
                config
            )

            yield {
                'images': images,
                'clearsky': clearsky,
            }, targets

    return create_generator
