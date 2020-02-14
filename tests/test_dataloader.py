import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
import datetime
import functools
import typing
import pytest
import json
import pickle
import os

from dataloader.dataloader import *


@pytest.fixture
def user_config():
    path = os.path.join(os.getcwd(), 'data/admin_cfg.json')
    assert os.path.isfile(path), f"invalid user config file: {path}"
    with open(path, "r") as fd:
        user_config = json.load(fd)
    return user_config


@pytest.fixture
def dataframe(user_config):
    path = os.path.join(os.getcwd(), user_config["dataframe_path"])
    assert os.path.isfile(path), f"invalid dataframe file: {path}"
    with open(path, 'rb') as f:
        x = pickle.load(f)
    x.loc[:, 'hdf5_16bit_path'] = os.path.join(os.getcwd(), 'data', '2015.02.19.0800.h5')
    return x


@pytest.fixture
def target_datetimes(user_config):
    return [datetime.datetime.fromisoformat(d) for d in user_config["target_datetimes"]]


@pytest.fixture
def station(user_config, config):
    station = {}
    station[config.station] = user_config["stations"][config.station]
    return station


@pytest.fixture
def target_time_offsets(user_config):
    return [pd.Timedelta(d).to_pytimedelta() for d in user_config["target_time_offsets"]]


@pytest.fixture
def config_dict():
    return {
        "station": "BND",
        "epoch": 15,
        "dataset_size": 1000,
        "seq_len": 6,
        "batch_size": 2,
        "model": 'lrcn',
        "channels": ["ch1", "ch2", "ch3", "ch4", "ch6"],
        "train": 1,
    }


@pytest.fixture
def config(config_dict):
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    config = Namespace(
        station=config_dict["station"],
        epoch=config_dict["epoch"],
        dataset_size=config_dict["dataset_size"],
        seq_len=config_dict["seq_len"],
        batch_size=config_dict["batch_size"],
        model=config_dict["model"],
        channels=config_dict["channels"],
        train=config_dict["train"],
    )
    return config


@pytest.fixture
def config_real(config):
    config.real = 1
    config.crop_size = 0
    return config


@pytest.fixture
def config_synthetic(config):
    config.real = 0
    config.crop_size = 60
    return config


@pytest.fixture
def target_stations(user_config):
    return user_config['stations']


def test_dataloader_real(dataframe, target_datetimes,
                         station, target_time_offsets,
                         config_real, target_stations):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real, target_stations
                            )
    assert isinstance(dl, tf.data.Dataset)


def test_dataloader_synthetic(dataframe, target_datetimes,
                              station, target_time_offsets,
                              config_synthetic, target_stations):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_synthetic, target_stations
                            )
    assert isinstance(dl, tf.data.Dataset)


def test_dataloader_real_img_ndims(dataframe, target_datetimes,
                                   station, target_time_offsets,
                                   config_real, target_stations):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real, target_stations
                            )
    for img, tgt in dl:
        break
    assert img.ndim == 5, f"{config_real}"


def test_dataloader_real_tgt_ndims(dataframe, target_datetimes,
                                   station, target_time_offsets,
                                   config_real, target_stations):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real, target_stations
                            )
    for img, tgt in dl:
        break
    assert tgt.ndim == 2


def test_dataloader_synthetic_img_ndims(dataframe, target_datetimes,
                                        station, target_time_offsets,
                                        config_synthetic, target_stations):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_synthetic, target_stations
                            )
    for img, tgt in dl:
        break
    assert img.ndim == 5


def test_dataloader_synthetic_tgt_ndims(dataframe, target_datetimes,
                                        station, target_time_offsets,
                                        config_synthetic, target_stations):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_synthetic, target_stations
                            )
    for img, tgt in dl:
        break
    assert tgt.ndim == 2
