def get_ghi_target():
    pass


def get_images():
    pass


def create_data_generator():
    pass


def get_random_trajectory(batch_size, seq_len, crop_size):
    frame_size = crop_size ** 2


class SyntheticMNISTDataload(object):
    """
    """

    def __init__(self, opts):
        super().__init__()

        self.batch_size = opts.seq_len
