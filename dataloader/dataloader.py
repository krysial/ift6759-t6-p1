import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
import datetime
import functools
import typing

from dataloader.get_crop_size import get_crop_size
from dataloader.dataset_processing import *
from dataloader.real import create_data_generator
from dataloader.get_station_px_center import get_station_px_center


def prepare_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]) -> tf.data.Dataset:
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

    generator = create_data_generator(
        dataframe=dataframe,
        target_datetimes=target_datetimes,
        station=station,
        target_time_offsets=target_time_offsets,
        config=config
    )

    # output_shapes = (seq_len, channels, height, width)
    data_loader = tf.data.Dataset.from_generator(
        generator, ({
            'images': tf.float32,
            'clearsky': tf.float32,
        }, tf.float32)
    )

    # Ankur not sure abou the below. Repeat is a hack, we need to see what's wrong

    # data_loader = data_loader.batch(config['batch_size']).repeat()
    # data_loader = data_loader.map(transposing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # if config['crop_size'] != 80:
    #     cropper = presaved_crop(config=config)
    #     data_loader = data_loader.map(cropper, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    data_loader = data_loader.map(normalize_station_GHI, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data_loader = data_loader.map(normalize_CLEARSKY_GHI, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Final step of data loading pipeline: Return the dataset loading object
    return data_loader
