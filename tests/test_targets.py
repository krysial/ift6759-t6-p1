from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np

from dataloader.get_GHI_targets import get_GHI_targets


# Set arguments
@pytest.fixture
def df():
    df_path = './data/dummy_test_catalog.pkl'
    return pd.read_pickle(df_path)


@pytest.fixture
def datetimes():
    datetimes = ["2015-01-01T12:45:00", "2015-12-31T22:00:00"]
    return [datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in datetimes]


@pytest.fixture
def station():
    return {'DRA': (0, 0, 0)}


@pytest.fixture
def offsets():
    offsets = ["P0DT0H0M0S", "P0DT1H0M0S", "P0DT3H0M0S", "P0DT6H0M0S"]
    return [timedelta(hours=int(x[4])) for x in offsets]


@pytest.fixture
def config():
    return {'target_name': 'GHI', 'target_past_len': 1, 'target_past_interval': 15}


# With offsets, target array should be of size [#datetimes, #offsets]
def test_shape_offsets(df, datetimes, station, offsets, config):
    targets = get_GHI_targets(df, datetimes, station, offsets, config)
    assert targets.shape == (len(datetimes), len(offsets))


# With sequence of images, target array should be of size [#datetimes, sequence length]
def test_shape_sequence(df, datetimes, station):
    offsets = [timedelta()]
    config = {'target_name': 'CLOUDINESS', 'target_past_len': 10, 'target_past_interval': 30}
    targets = get_GHI_targets(df, datetimes, station, offsets, config)
    assert targets.shape == (len(datetimes), config['target_past_len'])


# Target array should be numpy.ndarray and should only contain numpy.float64
def test_target_type(df, datetimes, station, offsets, config):
    targets = get_GHI_targets(df, datetimes, station, offsets, config)
    assert isinstance(targets, np.ndarray)
    assert all([isinstance(x, np.float64) for x in targets.flatten()])
