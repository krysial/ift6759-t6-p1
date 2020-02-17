import pytest
import datetime as dt
import pandas as pd

from dataloader import get_raw_images

DT_FORMAT = '%Y-%m-%dT%H:%M:%S'
h5_test_file = './data/2015.02.19.0800.h5'
app_config = {
    "no_of_temporal_seq": 5,
    "channels": ["ch1", "ch2", "ch3", "ch4", "ch6"],
    "goes13_dataset": 'hdf516',
    "target_datetimes": [
        "2015-02-18T12:45:00"
    ]
}


@pytest.fixture
def catalog():
    dataframe_path = "./data/dummy_test_catalog.pkl"
    catalog = pd.read_pickle(dataframe_path)
    return catalog


def test_tensor_output_shape(catalog):
    batch_of_datetimes = [dt.datetime.strptime(i, DT_FORMAT) for i in app_config['target_datetimes']]

    catalog.loc[:, 'hdf5_16bit_path'] = h5_test_file  # replacing real path for local file path
    output_tensor = get_raw_images.get_raw_images(catalog, batch_of_datetimes, app_config)
    assert output_tensor.shape == (1, 5, 5, 650, 1500), 'Output tensor should have correct dimensions'
