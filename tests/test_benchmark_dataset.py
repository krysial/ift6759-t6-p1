import pytest
import time
import tensorflow as tf

from dataloader.synthetic import create_synthetic_generator, Options

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

tf.executing_eagerly()


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    return time.perf_counter() - start_time


@pytest.fixture
def create_generator():
    def _create_generator():
        return create_synthetic_generator(opts)

    return _create_generator


@pytest.mark.skip(reason="performance test")
def test_vanilla_synthetic_benchmark(create_generator):
    generator = create_generator()
    data_loader = tf.data.Dataset.from_generator(
        generator,
        (tf.int64, tf.int64),
        args=[10],
    )
    assert benchmark(data_loader) <= 10
    assert generator is not None


@pytest.mark.skip(reason="performance test")
def test_prefetch_synthetic_benchmark(create_generator):
    generator = create_generator()
    data_loader = tf.data.Dataset.from_generator(
        generator,
        (tf.int64, tf.int64),
        args=[50],
    )
    b1 = benchmark(data_loader)

    data_loader_2 = tf.data.Dataset.from_generator(
        generator,
        (tf.int64, tf.int64),
        args=[50],
    ).prefetch(50)
    b2 = benchmark(data_loader_2)

    assert b1 <= 25
    assert b2 <= b1
