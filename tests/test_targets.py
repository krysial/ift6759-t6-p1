from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np

from dataloader.get_GHI_targets import get_GHI_targets


# Set arguments
@pytest.fixture
def df():
    df_path = '/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl'
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
    return {}


# Target array should be of size [#datetimes x #offsets]
def test_target_shape(df, datetimes, station, offsets, config):
    targets = get_GHI_targets(df, datetimes, station, offsets, config)
    assert targets.shape == (len(datetimes), len(offsets))


# Target array should be numpy.ndarray and should only contain numpy.float64
def test_target_type(df, datetimes, station, offsets, config):
    targets = get_GHI_targets(df, datetimes, station, offsets, config)
    assert isinstance(targets, np.ndarray)
    assert all([isinstance(x, np.float64) for x in targets.flatten()])
