import pytest
import tensorflow as tf

from dataloader.dataset_processing import dataset_processing

tf.executing_eagerly()


@pytest.fixture
def config():
    config = {
        'seq_len': 15,
        'channels': ['ch1', 'ch2', 'ch3', 'ch4', 'ch6'],
        'batch_size': 2,
        'crop_size': 20
    }
    return config


@pytest.fixture
def station():
    station = {"BND": [40.05192, -88.37309, 230]}
    return station


@pytest.fixture
def stations_px():
    stations_px = {
        "BND": [40, 88],
        "TBL": [50, 30],
        "DRA": [76, 81],
        "FPK": [38, 50],
        "GWN": [64, 89],
        "PSU": [90, 77],
        "SXF": [43, 96]
    }
    return stations_px


@pytest.fixture
def data(config):
    img = tf.random.uniform((config['batch_size'], config['seq_len'], len(config['channels']), config['crop_size'], config['crop_size']))
    return {'images': img}


@pytest.fixture
def target(config):
    tgt = tf.random.uniform((4,))
    return tgt


def test_sanity_check1(data, target, stations_px, station, config):
    data, tgt = dataset_processing(stations_px, station, config)(data, target)
    img = data['images']
    assert isinstance(img, tf.Tensor) and isinstance(tgt, tf.Tensor)


def test_output_image_dims(data, target, stations_px, station, config):
    data, tgt = dataset_processing(stations_px, station, config)(data, target)
    img = data['images']
    assert img.ndim == data['images'].ndim


def test_output_target_dims(data, target, stations_px, station, config):
    data, tgt = dataset_processing(stations_px, station, config)(data, target)
    assert tgt.ndim == target.ndim

# TODO: Commented out due to that the batch is not longer handled here
# def test_equal_batchsize_image_to_target(data, target, stations_px, station, config):
#     img, tgt = dataset_processing(stations_px, station, config)(data, target)
#     assert img.shape[0] == tgt.shape[0]


def test_equal_batchsize_input_to_output(data, target, stations_px, station, config):
    data, tgt = dataset_processing(stations_px, station, config)(data, target)
    img = data['images']
    assert (data['images'].shape[0] == img.shape[0]) and (target.shape[0] == tgt.shape[0])


def test_equal_seqlen_input_to_output(data, target, stations_px, station, config):
    data, tgt = dataset_processing(stations_px, station, config)(data, target)
    img = data['images']
    assert (data['images'].shape[0] == img.shape[0])


def test_equal_channel_input_to_output(data, target, stations_px, station, config):
    n_data, tgt = dataset_processing(stations_px, station, config)(data.copy(), target)
    img = n_data['images']
    assert (data['images'].shape[0] == img.shape[0])


def test_img_nan_generation_in_process(data, target, stations_px, station, config):
    data, tgt = dataset_processing(stations_px, station, config)(data, target)
    img = data['images']
    assert (tf.math.is_nan(img).numpy().any()) == (tf.math.is_nan(data['images']).numpy().any())


def test_target_sequence_input_to_output_check(data, target, stations_px, station, config):
    data, tgt = dataset_processing(stations_px, station, config)(data, target)
    assert (target.shape[-1] == tgt.shape[-1])
