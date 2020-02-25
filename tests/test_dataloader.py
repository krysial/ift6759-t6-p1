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
from dataloader.synthetic_dataloader \
    import prepare_dataloader as s_prepare_dataloader


@pytest.fixture
def admin_config():
    path = os.path.join(os.getcwd(), "data/admin_cfg.json")
    assert os.path.isfile(path), f"invalid user config file: {path}"
    with open(path, "r") as fd:
        admin_config = json.load(fd)
    return admin_config


@pytest.fixture
def dataframe(admin_config):
    path = os.path.join(os.getcwd(), admin_config["dataframe_path"])
    assert os.path.isfile(path), f"invalid dataframe file: {path}"
    with open(path, "rb") as f:
        x = pickle.load(f)
    x.loc[:, "hdf5_16bit_path"] = os.path.join(os.getcwd(), "data", "2015.02.19.0800.h5")
    return x


@pytest.fixture
def target_datetimes(admin_config):
    return [(datetime.datetime.fromisoformat(d), 0) for d in admin_config["target_datetimes"]]


@pytest.fixture
def station(admin_config):
    station = {}
    station["BND"] = admin_config["stations"]["BND"]
    return station


@pytest.fixture
def target_time_offsets(admin_config):
    return [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]


@pytest.fixture
def config():
    return {
        "batch_size": 2,
        "channels": ["ch1", "ch2", "ch3", "ch4", "ch6"],
        "goes13_dataset": "hdf516",
        "crop_size": 40,
        "seq_len": 6,
        "target_past_len": 1,
        "target_name": "GHI",
        "model": "lrcn",
        "cache_data_path": "data/cache",
        "input_past_interval": 15,
    }


@pytest.fixture
def config_real(config):
    config["real"] = 1
    config["crop_size"] = 0
    return config


@pytest.fixture
def config_synthetic(config):
    config["real"] = 0
    config["crop_size"] = 60
    return config


@pytest.fixture
def target_stations(admin_config):
    return admin_config["stations"]


def test_dataloader_real(dataframe, target_datetimes,
                         station, target_time_offsets,
                         config_real,
                         # target_stations
                         ):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real,
                            # target_stations
                            )
    assert isinstance(dl, tf.data.Dataset)


def test_dataloader_synthetic(dataframe, target_datetimes,
                              station, target_time_offsets,
                              config_synthetic,
                              # target_stations
                              ):
    dl = s_prepare_dataloader(
        dataframe, target_datetimes, station, target_time_offsets,
        config_synthetic
    )
    assert isinstance(dl, tf.data.Dataset)


def test_dataloader_real_img_ndims(dataframe, target_datetimes,
                                   station, target_time_offsets,
                                   config_real,
                                   # target_stations
                                   ):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real,
                            # target_stations
                            )
    for data, tgt in dl:
        break
    assert data['images'].ndim == 5, f"{config_real}"


def test_dataloader_real_tgt_ndims(dataframe, target_datetimes,
                                   station, target_time_offsets,
                                   config_real,
                                   # target_stations
                                   ):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real,
                            # target_stations
                            )
    for img, tgt in dl:
        break
    assert tgt.ndim == 2


def test_dataloader_synthetic_img_ndims(dataframe, target_datetimes,
                                        station, target_time_offsets,
                                        config_synthetic,
                                        # target_stations
                                        ):
    dl = s_prepare_dataloader(
        dataframe, target_datetimes, station, target_time_offsets,
        config_synthetic
    )
    for data, tgt in dl:
        break
    assert data['images'].ndim == 5


def test_dataloader_synthetic_tgt_ndims(dataframe, target_datetimes,
                                        station, target_time_offsets,
                                        config_synthetic,
                                        # target_stations
                                        ):
    dl = s_prepare_dataloader(
        dataframe, target_datetimes, station, target_time_offsets,
        config_synthetic
    )
    for img, tgt in dl:
        break
    assert tgt.ndim == 2


def test_dataloader_real_batch_size_check(dataframe, target_datetimes,
                                          station, target_time_offsets,
                                          config_real,
                                          # target_stations
                                          ):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real,
                            # target_stations
                            )
    for data, tgt in dl:
        break
    assert data['images'].shape[0] == config_real["batch_size"]
    assert tgt.shape[0] == config_real["batch_size"]


def test_dataloader_synthetic_batch_size_check(dataframe, target_datetimes,
                                               station, target_time_offsets,
                                               config_synthetic,
                                               # target_stations
                                               ):
    dl = s_prepare_dataloader(
        dataframe, target_datetimes, station, target_time_offsets,
        config_synthetic
    )

    for data, tgt in dl:
        break
    assert data['images'].shape[0] == config_synthetic["batch_size"]
    assert tgt.shape[0] == config_synthetic["batch_size"]


def test_dataloader_real_seq_len_check(dataframe, target_datetimes,
                                       station, target_time_offsets,
                                       config_real,
                                       # target_stations
                                       ):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real,
                            # target_stations
                            )
    for data, tgt in dl:
        break
    assert data['images'].shape[1] == config_real["seq_len"]


def test_dataloader_synthetic_seq_len_check(dataframe, target_datetimes,
                                            station, target_time_offsets,
                                            config_synthetic,
                                            # target_stations
                                            ):
    dl = s_prepare_dataloader(
        dataframe, target_datetimes, station, target_time_offsets,
        config_synthetic
    )

    for data, tgt in dl:
        break
    assert data['images'].shape[1] == config_synthetic["seq_len"]


def test_dataloader_real_channel_len_check(dataframe, target_datetimes,
                                           station, target_time_offsets,
                                           config_real,
                                           # target_stations
                                           ):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real,
                            # target_stations
                            )
    for data, tgt in dl:
        break
    assert data['images'].shape[-1] == len(config_real["channels"])


def test_dataloader_synthetic_channel_len_check(
    dataframe, station, target_datetimes,
    target_time_offsets, config_synthetic
):
    dl = s_prepare_dataloader(
        dataframe, target_datetimes, station, target_time_offsets,
        config_synthetic
    )

    for data, tgt in dl:
        break
    assert data['images'].shape[-1] == len(config_synthetic["channels"])


def test_dataloader_real_clearsky_ghi_size_check(dataframe, target_datetimes,
                                                 station, target_time_offsets,
                                                 config_real,
                                                 # target_stations
                                                 ):
    dl = prepare_dataloader(dataframe, target_datetimes,
                            station, target_time_offsets,
                            config_real,
                            # target_stations
                            )
    for data, tgt in dl:
        break
    assert data['clearsky'].shape[0] == config_real["batch_size"]
    assert data['clearsky'].shape[1] == len(target_time_offsets)
