import pytest
import tensorflow as tf

from utils.synthetic import create_mnist_generator, Options

tf.executing_eagerly()

opts = Options(100, 50, 28, 12, 5)


@pytest.fixture
def create_generator():
    def _create_generator():
        return create_mnist_generator(opts)

    return _create_generator


def test_sanity_check(create_generator):
    generator = create_generator()
    assert generator is not None


def test_datasetloader(create_generator):
    generator = create_generator()

    data_loader = tf.data.Dataset.from_generator(
        generator,
        (tf.int64, tf.int64)
    )

    assert data_loader is not None

    image, ghi = next(iter(data_loader))

    assert tf.debugging.is_numeric_tensor(image)
    assert tf.debugging.is_numeric_tensor(ghi)
