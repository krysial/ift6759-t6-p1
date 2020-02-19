from collections import namedtuple
from itertools import zip_longest
import pandas as pd
import numpy as np

from utils.synthetic_mnist_generator import SyntheticMNISTGenerator, \
    Options as SyntheticMNISTGeneratorOptions

from utils.synthetic_ghi_processor import SyntheticGHIProcessor, \
    Options as SyntheticGHIProcessorOptions

try:
    import pydevd
    DEBUGGING = True
except ImportError:
    DEBUGGING = False


Options = namedtuple(
    'SyntheticGeneratorOptions',
    [
        'image_size',
        'digit_size',
        'num_channels',
        'seq_len',
        'step_size',
        'lat',
        'lon',
        'alt',
        'offsets'
    ]
)


def create_synthetic_generator(opts):
    """
    A generator that prepares synthetic data for the models.
    They are a multi layer moving MNIST characters that
    bounce off the edges of the images.

    Args:
        opts: protobuf that contains the options of the synthetic data,
        see SyntheticMNISTGeneratorOptions for options
    """
    def create_generator():
        # This is needed to allow debugging
        # tf.data.dataset iterates in a separate C thread
        # preventing the debugger to be called.
        if DEBUGGING:
            pydevd.settrace(suspend=False)

        def convert_to_index(offset):
            offset_delta = pd.Timedelta(offset)
            return int(offset_delta.seconds / (60 * 15))

        offsets = list(map(convert_to_index, opts.offsets))

        mnist_opts = SyntheticMNISTGeneratorOptions(
            opts.image_size,
            opts.digit_size,
            opts.num_channels,
            opts.seq_len + offsets[-1],
            opts.step_size
        )

        ghi_opts = SyntheticGHIProcessorOptions(
            opts.lat,
            opts.lon,
            opts.alt,
            offsets
        )

        ghi_processor = SyntheticGHIProcessor(ghi_opts)
        mnist_generator = SyntheticMNISTGenerator(mnist_opts)

        def dataset_data_broker(data):
            ghi, clearsky_ghi = ghi_processor.processData(data)

            return (
                data[offsets[-1]:, :, :, :],
                clearsky_ghi,
                ghi
            )

        for i, (seq, clearsky, ghi) in enumerate(map(
            dataset_data_broker,
            mnist_generator
        )):
            if DEBUGGING:
                pydevd.settrace(suspend=False)

            yield {
                'images': seq,
                'clearsky': clearsky
            }, ghi

    return create_generator
