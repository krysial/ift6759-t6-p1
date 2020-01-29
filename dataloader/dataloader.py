import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
import datetime
import typing

from get_GHI_targets import get_GHI_targets
from get_raw_images import get_raw_images
from get_crop_frame_size import get_crop_frame_size


def prepare_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],) -> tf.data.Dataset:

"""
A function to prepare the dataloader

Args:
    dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset).
    batch_of_datetimes: a batch of timestamps that is required by the data loader to provide targets for the model.
    station: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
    target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
    config: configuration dictionary holding any extra parameters that might be required for tuning purposes.
    target_stations: A dictionary of all target stations with respective coordinates and elevation

Returns:
    data_loader: A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``. The shape of the tf.data.Dataset should be ([None, temporal_seq, 5, crop_size, crop_size], [None, 4])
"""


def create_data_generator(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],):
    """
   A function to create a generator to yield data to the dataloader
   """

    for i in range(0, len(target_datetimes), batch_size):
        batch_of_datetimes = target_datetimes[i:i+batch_size]
        targets = get_GHI_targets(
            dataframe, batch_of_datetimes, station, target_time_offsets, config)
        images = get_raw_images(dataframe, batch_of_datetimes, config)
        yield images, targets

    if config['crop_size'] == None:
        config['crop_size'] = get_crop_frame_size(dataframe, target_stations)

    generator = create_data_generator(
        dataframe, target_datetimes, station, target_time_offsets, config)
    data_loader = tf.data.Dataset.from_generator(
        generator, (tf.float32, tf.float32))

    return data_loader
