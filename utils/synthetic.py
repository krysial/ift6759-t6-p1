from collections import namedtuple
import os
import sys
import h5py

dirname = os.path.dirname(__file__)


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
    [
        'batch_size',
        'image_size',
        'digit_size',
        'seq_len'
    ]
)


class SyntheticMNIST(object):
    def __init__(self):
        pass


def syntheticMNISTGenerator(opts):
    """
    A generator that prepares synthetic data for the models.
    They are a multi layer moving MNIST characters that
    bounce off the edges of the images.

    Args:
        opts: protobuf that contains the options of the synthetic data
    """
    try:
        f = h5py.File(
            os.path.abspath(
                os.path.join(dirname, '../data/mnist.h5')
            )
        )
    except Exception:
        print('Please set the correct path to MNIST dataset')
        sys.exit()

    def getRandomTrajectory():
        batch_size = opts.batch_size
        length = opts.seq_len
        canvas_size = opts.image_size - opts.digit_size

        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        for i in xrange(length):
            # Take a step along velocity.
            y += v_y * length
            x += v_x * length

            # Bounce off edges.
            for j in xrange(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generator():
        while True:
            yield 0

    return generator
