import typing
import datetime as dt
import json

import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import utils

IMAGE_HEIGHT = 650
IMAGE_WIDTH = 1500
DT_FORMAT = '%Y-%m-%dT%H:%M:%S'
GOES13_DS = {
    'hdf516': ['hdf5_16bit_path', 'hdf5_16bit_offset'],
    'hdf508': ['hdf5_8bit_path', 'hdf5_8bit_offset']
}


def get_preprocessed_images(
        dataframe: pd.DataFrame,
        datetimes: typing.List[dt.datetime],
        config: typing.Dict[typing.AnyStr, typing.Any]) -> np.ndarray:

    channels = config['channels']
    seqs = config['seq_len']

    frames = fetch_preprocessed_frames(dataframe, datetimes, config)

    assert frames.shape == (len(datetimes), seqs, len(channels), config['crop_size'], config['crop_size'])
    return frames


def get_raw_images(
        dataframe: pd.DataFrame,
        datetimes: typing.List[dt.datetime],
        config: typing.Dict[typing.AnyStr, typing.Any],) -> np.ndarray:
    """
    A function to get the raw input images as taken by the satellite

    Args:
        dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset).
        datetimes: a batch of timestamps that is required by the data loader to provide raw uncropped versions of
                            the satellite images for the model.
        config: configuration dictionary holding any extra parameters that might be required for tuning purposes.

    Returns:
        A numpy array of shape (batch_size, temporal_seq, channels, length, width)
    """

    channels = config['channels']
    seqs = config['seq_len']
    goes13_i_paths = get_frames_location(dataframe, datetimes, seqs, config['goes13_dataset'])
    frames = fetch_frames(datetimes, goes13_i_paths, channels, seqs)

    assert frames.shape == (len(datetimes), seqs, len(channels), IMAGE_HEIGHT, IMAGE_WIDTH)
    return frames

# auxiliary functions


def read_conf_file(path):
    with open(path) as f:
        config = json.load(f)
    return config


def get_frames_location(dataframe, datetimes, seqs, dataset):

    columns = GOES13_DS[dataset] if dataset else GOES13_DS['hdf516']
    offset = 15
    dt_seqs = []

    for i, datetime in enumerate(datetimes):
        for j in range(seqs):
            dt_seqs.append({
                'datetime': datetime - dt.timedelta(minutes=(offset * j)),
                'position': (i, j)
            })

    df = pd.DataFrame(dt_seqs)
    seq_datetimes = df['datetime']
    df['path'] = dataframe.reindex(seq_datetimes)[columns[0]].to_list()
    df['offset'] = dataframe.reindex(seq_datetimes)[columns[1]].to_list()

    return df.dropna()


def fetch_preprocessed_frames(dateframe, frames_df, config):
    dataset = config['goes13_dataset']
    path_offset_columns = GOES13_DS[dataset] if dataset else GOES13_DS['hdf516']

    output = np.zeros((
        config['batch_size'],
        config['seq_len'],
        len(config['channels']),
        config['crop_size'],
        config['crop_size']
    ))

    cache = {}
    for batch, (start_datetime, station_id) in enumerate(frames_df):
        times = pd.date_range(
            start=start_datetime,
            periods=config['seq_len'],
            freq='15Min',
            tz='utc'
        )

        paths_groups = dateframe[dateframe.index.isin(times)].groupby(path_offset_columns, sort=False)

        for seq, ((path, offset), group) in enumerate(paths_groups):
            filename = os.path.basename(os.path.normpath(path))
            fullpath = config['cache_data_path'] + str(config['crop_size']) + '/' + filename

            if os.path.exists(fullpath) and filename not in cache:
                cache[filename] = np.memmap(
                    fullpath,
                    dtype='float32',
                    mode='r',
                    shape=(
                        len(config['channels']),
                        96,
                        7,  # hardcoded number of stations
                        config['crop_size'],
                        config['crop_size']
                    )
                )

            if filename in cache:
                output[batch, seq] = cache[filename][:, offset, station_id]

    return output


def fetch_frames(datetimes, frames_df, channels, seqs):
    output = np.empty((len(datetimes), seqs, len(channels), IMAGE_HEIGHT, IMAGE_WIDTH))

    paths_groups = frames_df.groupby('path', sort=False)

    for name, group in paths_groups:
        with h5py.File(name, "r") as h5_data:
            for index, row in group.iterrows():
                position = row['position']
                frame = row['offset']
                for c, channel in enumerate(channels):
                    channel_idx_data = utils.fetch_hdf5_sample(channel, h5_data, frame)
                    if channel_idx_data is None or channel_idx_data.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                        output[position[0], position[1], c] = 0
                    else:
                        output[position[0], position[1], c] = tf.keras.utils.normalize(channel_idx_data)

    return output


def show_frame(image):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='bone')
    plt.show()
