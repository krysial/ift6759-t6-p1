import pytest
import tensorflow as tf

from utils.synthetic import syntheticMNISTGenerator, Options


opts = Options(100, 12, 12, 12)


@pytest.fixture
def setup_generator():
    def _setup_generator():
        return syntheticMNISTGenerator(opts)

    return _setup_generator


def test_datasetloader(setup_generator):
    generator = setup_generator()

    data_loader = tf.data.Dataset.from_generator(
        generator,
        (tf.int64, tf.int64)
    )

    assert data_loader is not None


def test_answer(setup_generator):
    generator = setup_generator()
    assert generator is not None
