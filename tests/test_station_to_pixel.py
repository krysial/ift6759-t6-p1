from unittest import mock

import pytest
import pandas as pd

from dataloader import get_station_px_center

dataframe_path = "./data/dummy_test_catalog.pkl"
h5_test_file = './data/2015.02.19.0800.h5'

DT_FORMAT = '%Y-%m-%dT%H:%M:%S'
app_config = {
    "no_of_temporal_seq": 5,
    "channels": ["ch1", "ch2", "ch3", "ch4", "ch6"],
    "goes13_dataset": 'hdf516',
    "target_datetimes": [
        "2015-02-18T12:45:00"
    ],
    "stations": {
        "BND": [40.05192, -88.37309, 230],
        "TBL": [40.12498, -105.23680, 1689],
        "DRA": [36.62373, -116.01947, 1007],
        "FPK": [48.30783, -105.10170, 634],
        "GWN": [34.25470, -89.87290, 98],
        "PSU": [40.72012, -77.93085, 376],
        "SXF": [43.73403, -96.62328, 473]
    }
}


@pytest.fixture
def catalog():
    # replacing real path for local file path
    catalog_df = pd.read_pickle(dataframe_path)
    catalog_df.loc[:, 'hdf5_16bit_path'] = h5_test_file

    return catalog_df


def randint_mock(x, y):
    # the test h5 file is missing a lot of frames, forcing to point to a known one
    return 67


def test_lat_lon_coords(catalog):
    with mock.patch('numpy.random.randint', randint_mock):
        lats, lons = get_station_px_center.get_lats_longs_goes13_coords(catalog)
        assert len(lats) == 650, 'Incorrect size for lats'
        assert len(lons) == 1500, 'Incorrect size for lons'


def test_output_same_stations_size(catalog):
    stations = {k: (v[0], v[1], v[2]) for k, v in app_config['stations'].items()}

    with mock.patch('numpy.random.randint', randint_mock):
        stations_px_coordinates = get_station_px_center.get_station_px_center(catalog, stations)
        assert stations_px_coordinates.keys() == stations.keys(), 'Stations keys should be the same as input'
