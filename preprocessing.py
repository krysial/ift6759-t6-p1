import typing
import pandas as pd
import datetime
import json
import os

import h5py

from dataloader import  get_raw_images
from utils import utils

IMAGE_HEIGHT = 650
IMAGE_WIDTH = 1500


def create_data_generator(
        dataframe: pd.DataFrame,
        station: typing.Dict[
            typing.AnyStr,
            typing.Tuple[float, float, float]
        ]):
    """
    A function to create a generator to yield data to the dataloader
    """

    unique_paths = reversed(list(dataframe.groupby('hdf5_16bit_path').groups.keys()))

    for path in unique_paths:
        if os.path.isfile(path):
            print(path)
            with h5py.File(path, "r") as h5_data:
                for frame in range(96):
                    for channel in ['ch1', 'ch2', 'ch3', 'ch4', 'ch6']:
                        channel_idx_data = utils.fetch_hdf5_sample(channel, h5_data, frame)
                        if channel_idx_data is None or channel_idx_data.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                            channel_idx_data = 0



if __name__ == '__main__':
    dataframe_path = "/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl"

    catalog = pd.read_pickle(dataframe_path)
    with open('./data/admin_cfg.json') as f:
        config = json.load(f)
    stations = config['stations']

    create_data_generator(catalog, stations)