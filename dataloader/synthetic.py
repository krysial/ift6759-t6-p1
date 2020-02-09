from collections import namedtuple
from itertools import zip_longest

from utils.synthetic_mnist_generator import SyntheticMNISTGenerator, \
    Options as SyntheticMNISTGeneratorOptions

from utils.synthetic_ghi_processor import SyntheticGHIProcessor, \
    Options as SyntheticGHIProcessorOptions

try:
    import pydevd
    DEBUGGING = True
except ImportError:
    DEBUGGING = False

import time

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
        'alt'
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
    def create_generator(dataset_size=1000):
        # This is needed to allow debugging
        # tf.data.dataset iterates in a separate C thread
        # preventing the debugger to be called.
        if DEBUGGING:
            pydevd.settrace(suspend=False)

        mnist_opts = SyntheticMNISTGeneratorOptions(
            opts.image_size,
            opts.digit_size,
            opts.num_channels,
            opts.seq_len,
            opts.step_size
        )

        ghi_opts = SyntheticGHIProcessorOptions(
            opts.lat,
            opts.lon,
            opts.alt
        )

        ghi_processor = SyntheticGHIProcessor(ghi_opts)
        mnist_generator = SyntheticMNISTGenerator(mnist_opts)

        for i, (seq, ghi) in enumerate(map(
            lambda data: (data, ghi_processor.processData(data)),
            mnist_generator
        )):
            if DEBUGGING:
                pydevd.settrace(suspend=False)

            if i < dataset_size:
                yield seq, ghi
            else:
                break

    return create_generator
