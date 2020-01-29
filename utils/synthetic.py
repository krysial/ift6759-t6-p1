from collections import namedtuple


def get_ghi_target():
    pass


def get_images():
    pass


def create_data_generator():
    pass


def get_random_trajectory(batch_size, seq_len, crop_size):
    frame_size = crop_size ** 2


Options = namedtuple(
    'SyntheticMNISTGeneratorOptions',
    'batch_size seq_len'
)


class SyntheticMNISTGenerator(object):
    """
    A class that prepares synthetic data for the models. They are a multi layer
    moving MNIST characters that bounce off the edges of the images.

    Args:
        opts: protobuf that contains the options of the synthetic data
    """

    def __init__(self, opts):
        super().__init__()

        self.batch_size = opts.batch_size
        self.seq_len = opts.seq_len

    def __iter__(self):
        pass
