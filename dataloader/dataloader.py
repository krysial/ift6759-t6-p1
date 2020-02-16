import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
import datetime
import functools
import typing

from dataloader.get_crop_size import get_crop_size
from dataloader.dataset_processing import dataset_processing
from dataloader.real import create_data_generator
from dataloader.get_station_px_center import get_station_px_center
from dataloader.synthetic import create_synthetic_generator, Options


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
    config_dict = config if type(config) == dict else vars(config)
    if not config_dict['real']:
        opts = Options(
            image_size=config_dict['crop_size'],
            digit_size=28,
            num_channels=len(config_dict['channels']),
            seq_len=config_dict['seq_len'],
            step_size=0.3,
            lat=station[config_dict['station']][0],
            lon=station[config_dict['station']][1],
            alt=station[config_dict['station']][2],
            offsets=target_time_offsets
        )
        generator = create_synthetic_generator(opts)
        data_loader = tf.data.Dataset.from_generator(
            generator, (tf.float32, tf.float32)
        ).batch(config_dict['batch_size'])
    else:
        # First step in the data loading pipeline:
        # A generator object to retrieve a inputs resources and their targets
        config_dict['target_datetimes'] = target_datetimes
        config_dict['goes13_dataset'] = 'hdf516'

        generator = create_data_generator(
            dataframe=dataframe,
            target_datetimes=target_datetimes,
            station=station,
            target_time_offsets=target_time_offsets,
            config=config_dict
        )

        data_loader = tf.data.Dataset.from_generator(
            generator, (tf.float32, tf.float32))

        # Second step: Estimate/Calculate station
        # coordinates on image and crop area dimensions
        stations_px = get_station_px_center(dataframe, station)
        if config_dict['crop_size'] == 0:
            config_dict['crop_size'] = get_crop_size(stations_px, data_loader)

        # Third step: Processing using map (cropping for stations)
        data_processor = dataset_processing(
            stations_px=stations_px,
            station=station,
            config=config_dict
        )

        data_loader = data_loader.map(
            data_processor
        ).cache(
            filename='/project/cq-training-1/project1/teams/team06/cache'
        )

    # Final step of data loading pipeline: Return the dataset loading object
    return data_loader
