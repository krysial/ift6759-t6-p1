import typing
import pandas as pd
import datetime
import json
import os

import h5py

from dataloader import get_raw_images
from utils import utils

import functools

IMAGE_HEIGHT = 650
IMAGE_WIDTH = 1500


def create_data_generator(
        dataframe: pd.DataFrame,
        stations: typing.Dict[
            typing.AnyStr,
            typing.Tuple[float, float, float]
        ]):
    """
    A function to create a generator to yield data to the dataloader
    """

    unique_paths = reversed(list(dataframe.groupby('hdf5_16bit_path').groups.keys()))
    stations_px_center = []

    for path in unique_paths:
        if os.path.isfile(path):
            with h5py.File(path, "r") as h5_data:
                for frame in range(96):
                    for channel in ['ch1', 'ch2', 'ch3', 'ch4', 'ch6']:
                        channel_idx_data = utils.fetch_hdf5_sample(channel, h5_data, frame)
                        if channel_idx_data is None or channel_idx_data.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                            channel_idx_data = 0
                        else:
                            if not stations_px_center:
                                lats = utils.fetch_hdf5_sample("lat", h5_data, 0)
                                lons = utils.fetch_hdf5_sample("lon", h5_data, 0)

                                def red_coords_to_px(aggre, s):
                                    lat, lon, _ = stations[s]
                                    px_lat = len(lats) * ((lat - lats[0]) / (lats[-1] - lats[0]))
                                    px_lon = len(lons) * ((lon - lons[0]) / (lons[-1] - lons[0]))

                                    aggre[s] = (px_lat, px_lon)
                                    return aggre

                                stations_px_center = functools.reduce(red_coords_to_px, stations, {})
                            else:
                                pass

                        del lats
                        del lons
                        del channel_idx_data

                h5_data.close()


if __name__ == '__main__':
    dataframe_path = "/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl"

    catalog = pd.read_pickle(dataframe_path)
    with open('./data/admin_cfg.json') as f:
        config = json.load(f)

    stations = config['stations']
    create_data_generator(catalog, stations)
