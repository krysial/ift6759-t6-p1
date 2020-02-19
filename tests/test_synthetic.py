import pytest
import tensorflow as tf
import numpy as np

from dataloader.synthetic import create_synthetic_generator, Options

tf.executing_eagerly()

opts = Options(
    image_size=300,
    digit_size=28,
    num_channels=5,
    seq_len=12,
    step_size=0.5,
    lat=32.2,
    lon=-111,
    alt=700,
    offsets=["P0DT0H0M0S", "P0DT1H0M0S", "P0DT3H0M0S", "P0DT6H0M0S"]
)


@pytest.fixture
def create_generator():
    def _create_generator():
        np.random.seed(100)
        return create_synthetic_generator(opts)

    return _create_generator


def test_sanity_check(create_generator):
    generator = create_generator()
    assert generator is not None


def test_datasetloader_images(create_generator):
    generator = create_generator()

    data_loader = tf.data.Dataset.from_generator(
        generator, ({
            'images': tf.float32,
            'clearsky': tf.float32,
        }, tf.float32)
    )

    assert data_loader is not None

    data, _ = next(iter(data_loader))
    image = data['images']

    assert tf.debugging.is_numeric_tensor(image)
    assert image.numpy().shape == (12, 300, 300, 5)


def test_datasetloader_ghi(create_generator):
    generator = create_generator()

    data_loader = tf.data.Dataset.from_generator(
        generator, ({
            'images': tf.float32,
            'clearsky': tf.float32,
        }, tf.float32)
    )

    assert data_loader is not None

    _, ghi = next(iter(data_loader))

    assert ghi is not None
    assert np.array_equal(
        ghi.numpy(),
        np.array([0, 0, 0, 64.80217], dtype='float32')
    )
