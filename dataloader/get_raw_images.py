import pandas as pd
import json
import numpy as np

import typing
import datetime as dt
import h5py
import matplotlib.pyplot as plt

from utils import utils

image_height = 650
image_width = 1500
dt_format = '%Y-%m-%dT%H:%M:%S'


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
    seqs = config['no_of_temporal_seq']
    datetimes = [dt.datetime.strptime(i, dt_format) for i in datetimes]
    goes13_i_paths = get_frames_location(catalog, datetimes, seqs)
    frames = fetch_frames(goes13_i_paths, channels, seqs)

    assert frames.shape == (len(datetimes), seqs, len(channels), image_height, image_width)
    return frames

# auxiliary functions


def read_conf_file(path):
    with open(path) as f:
        config = json.load(f)
    return config


def get_frames_location(catalog, datetimes, seqs):
    # what to do when no frame for specific datetime (when no path?)?
    # what to do when no past dataframes entries e.g very beginning of the catalog???
    columns = ['hdf5_16bit_path', 'hdf5_16bit_offset']
    offset = 15
    output = []

    for datetime in datetimes:
        output.append(catalog.loc[[datetime - dt.timedelta(minutes=(offset * i)) for i in range(seqs)], columns])

    return output


def fetch_frames(images_path, channels, seqs):
    # should we scale images ?
    # TODO: v2 -> update to read same frames' files only once
    output = np.empty((len(images_path), seqs, len(channels), image_height, image_width))

    for i, seq_catalog in enumerate(images_path):
        for j, (path, frame) in enumerate(zip(seq_catalog.iloc[:, 0], seq_catalog.iloc[:, 1])):
            with h5py.File(path, "r") as h5_data:
                for k, channel in enumerate(channels):
                    channel_idx_data = utils.fetch_hdf5_sample(channel, h5_data, frame)
                    assert channel_idx_data is None or channel_idx_data.shape == (image_height, image_width), \
                        "the channels had an unexpected dimension"
                    output[i][j][k] = channel_idx_data

    return output


def show_frame(image):
    assert image.shape == (image_height, image_width), 'Unexpected frame shape'
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='bone')
    plt.show()


if __name__ == "__main__":
    app_config = {
        "no_of_temporal_seq": 2,
        "dataframe_path": "/project/cq-training-1/project1/data/dummy_test_catalog.pkl",
        "channels": ["ch1", "ch2", "ch3", "ch4", "ch6"],
        "target_datetimes": [
            "2015-01-01T12:45:00",
            "2015-01-01T19:00:00",
            "2015-01-02T13:30:00",
            "2015-01-02T19:45:00"
        ]
    }
    DATAFRAME_PATH = app_config['dataframe_path']
    catalog = pd.read_pickle(DATAFRAME_PATH)
    batch_of_datetimes = app_config['target_datetimes']

    raw_images = get_raw_images(catalog, batch_of_datetimes, app_config)

    show_frame(raw_images[1, 1, 0])
    print('Done!')