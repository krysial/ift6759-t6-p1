import typing
import pandas as pd
import datetime
import json
import os

import h5py
import numpy as np

from dataloader import get_raw_images
from utils import utils
from queue import Queue
from threading import Thread

import functools
import argparse

IMAGE_HEIGHT = 650
IMAGE_WIDTH = 1500

DATAFRAME_PATH = "/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl"
OUTPUT_PATH = "/project/cq-training-1/project1/teams/team06/cache/hdf5v5_16bit/"


class Worker(Thread):
    def __init__(self, queue, config):
        Thread.__init__(self)
        self.queue = queue
        self.stations_px_center = []
        self.px_offset = config.crop_size // 2
        self.config = config

        with open('./data/admin_cfg.json') as f:
            admin_config = json.load(f)
            f.close()

        self.stations = admin_config['stations']

    def run(self):
        while True:
            path = self.queue.get()
            try:
                self.worker_task(path)
            finally:
                self.queue.task_done()

    def worker_task(self, path):
        if os.path.isfile(path):
            with h5py.File(path, "r") as h5_data:
                file_data = []
                for offset in range(96):
                    images = []
                    for channel in ['ch1', 'ch2', 'ch3', 'ch4', 'ch6']:
                        channel_idx_data = utils.fetch_hdf5_sample(channel, h5_data, offset)
                        if channel_idx_data is None or channel_idx_data.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                            channel_idx_data = 0
                        else:
                            if not self.stations_px_center:
                                lats = utils.fetch_hdf5_sample("lat", h5_data, 0)
                                lons = utils.fetch_hdf5_sample("lon", h5_data, 0)

                                def red_coords_to_px(aggre, s):
                                    lat, lon, _ = self.stations[s]
                                    px_lat = len(lats) * ((lat - lats[0]) / (lats[-1] - lats[0]))
                                    px_lon = len(lons) * ((lon - lons[0]) / (lons[-1] - lons[0]))

                                    aggre[s] = (int(px_lat), int(px_lon))
                                    return aggre

                                self.stations_px_center = functools.reduce(red_coords_to_px, self.stations, {})
                                del lats
                                del lons

                            def crop(s):
                                center = self.stations_px_center[s]
                                px_x_ = center[0] - self.px_offset
                                px_x = center[0] + self.px_offset
                                px_y_ = center[1] - self.px_offset
                                px_y = center[1] + self.px_offset
                                return channel_idx_data[px_x_:px_x, px_y_:px_y]

                            images.append(list(map(crop, self.stations_px_center)))

                            del channel_idx_data

                    if images:
                        stations_images = np.transpose(np.array(images), [1, 2, 3, 0])
                    else:
                        stations_images = []

                    file_data.append(stations_images)

                result = np.array(file_data)
                save_path_base = OUTPUT_PATH + str(self.config.crop_size) + '/'

                if not(os.path.exists(save_path_base)):
                    os.mkdir(save_path_base)

                np.save(
                    save_path_base +
                    os.path.basename(os.path.normpath(path)),
                    result
                )

                del result
                del file_data
                h5_data.close()


def main(config, dataframe):
    queue = Queue()
    unique_paths = list(
        dataframe.groupby('hdf5_16bit_path').groups.keys()
    )[config.start_index:config.end_index]

    for x in range(20):
        worker = Worker(queue, config)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()

    for path in unique_paths:
        if path == "nan":
            print('Nan encountered')
        else:
            print('Queueing {}'.format(path))
            queue.put(path)

    queue.join()
    print('done')


if __name__ == '__main__':
    dataframe = pd.read_pickle(DATAFRAME_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-index",
        type=int,
        help="start dateframe index",
        default=0
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="end dateframe index",
        default=len(dataframe)
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        help="size of the crop frame",
        default=100
    )
    args = parser.parse_args()

    main(
        args,
        dataframe
    )
