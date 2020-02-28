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
        def batch_datetimes():
            for index in target_datetimes:
                row = dataframe.loc[index]
                if 'BND' in station:
                    yield [(index, [*station].index('BND'))]
                if 'TBL' in station:
                    yield [(index, [*station].index('TBL'))]
                if 'DRA' in station:
                    yield [(index, [*station].index('DRA'))]
                if 'FPK' in station:
                    yield [(index, [*station].index('FPK'))]
                if 'GWN' in station:
                    yield [(index, [*station].index('GWN'))]
                if 'PSU' in station:
                    yield [(index, [*station].index('PSU'))]
                if 'SXF' in station:
                    yield [(index, [*station].index('SXF'))]

        for batch in batch_datetimes():
            images = get_raw_images(
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


def create_data_generator_train(
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
            while True:
                sample = dataframe.sample()
                index = sample.index[0]
                row = sample.iloc[0]

                if row['BND_DAYTIME'] == 1 and 'BND' in station:
                    filtered_df.append((index, [*station].index('BND')))
                if row['TBL_DAYTIME'] == 1 and 'TBL' in station:
                    filtered_df.append((index, [*station].index('TBL')))
                if row['DRA_DAYTIME'] == 1 and 'DRA' in station:
                    filtered_df.append((index, [*station].index('DRA')))
                if row['FPK_DAYTIME'] == 1 and 'FPK' in station:
                    filtered_df.append((index, [*station].index('FPK')))
                if row['GWN_DAYTIME'] == 1 and 'GWN' in station:
                    filtered_df.append((index, [*station].index('GWN')))
                if row['PSU_DAYTIME'] == 1 and 'PSU' in station:
                    filtered_df.append((index, [*station].index('PSU')))
                if row['SXF_DAYTIME'] == 1 and 'SXF' in station:
                    filtered_df.append((index, [*station].index('SXF')))

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
