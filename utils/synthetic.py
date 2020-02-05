from collections import namedtuple
import os
import sys
import h5py
import numpy as np

try:
    import pydevd
    DEBUGGING = True
except ImportError:
    DEBUGGING = False


dirname = os.path.dirname(__file__)

Options = namedtuple(
    'SyntheticMNISTGeneratorOptions',
    [
        'image_size',
        'digit_size',
        'num_channels',
        'seq_len'
    ]
)


class SyntheticMNISTGenerator(object):
    def __init__(self, opts: Options):
        super().__init__()
        try:
            self.opts = opts
            f = h5py.File(
                os.path.abspath(
                    os.path.join(dirname, '../data/mnist.h5')
                )
            )
            self.data_ = f['train']['inputs'][()].reshape(-1, 28, 28)
            f.close()
            self.indices_ = np.arange(self.data_.shape[0])
            self.row_ = 0
            np.random.shuffle(self.indices_)
        except Exception:
            print('Please set the correct path to MNIST dataset')
            sys.exit()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def getRandomTrajectory(self):
        length = self.opts.seq_len
        canvas_size = self.opts.image_size - self.opts.digit_size

        # Initial position uniform random inside the box.
        y = np.random.rand()
        x = np.random.rand()

        # Choose a random velocity.
        theta = np.random.rand() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length))
        start_x = np.zeros((length))
        for i in range(length):
            # Take a step along velocity.
            y += v_y * length
            x += v_x * length

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y

            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)

    def video(self):
        start_y, start_x = self.getRandomTrajectory()

        data = np.zeros(
            (
                self.opts.seq_len,
                self.opts.image_size,
                self.opts.image_size
            ),
            dtype=np.float32
        )

        for n in range(self.opts.num_channels):
            # get random digit from dataset
            ind = self.indices_[self.row_]
            self.row_ += 1
            if self.row_ == self.data_.shape[0]:
                self.row_ = 0
                np.random.shuffle(self.indices_)
            digit_image = self.data_[ind, :, :]

            # generate video
            for i in range(self.opts.seq_len):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.opts.digit_size
                right = left + self.opts.digit_size
                data[i, top:bottom, left:right] = self.overlap(
                    data[i, top:bottom, left:right],
                    digit_image
                )

        return data.reshape(-1)

    def next(self):
        return self.video(), 3.0


def create_mnist_generator(opts):
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

        return SyntheticMNISTGenerator(opts)

    return create_generator
