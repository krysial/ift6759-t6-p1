import tensorflow as tf
import typing


try:
    import pydevd
    DEBUGGING = True
except ImportError:
    DEBUGGING = False


def dataset_processing(
        stations_px: typing.Dict[typing.AnyStr, typing.Tuple[float, float]],
        station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        config: typing.Dict[typing.AnyStr, typing.Any],):
    """
    A mapping lambda function to define the processing of the dataset once they are loaded

    Args:
        image_tensor: A tensor element of images from dataset object.
        target_tensor: A tensor element of target from dataset object.
        stations_px: a map of station names with their pixel coordinates on image.
        station: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        config: configuration dictionary holding any extra parameters that might be required for tuning purposes.

    Returns:
        image_tensor_: image tensor after processing operation.
        target_tensor_: target tensor after processing operation.
    """
    if DEBUGGING:
        pydevd.settrace(suspend=False)

    # Function to crop image
    def crop(image_tensor, keys):
        center = stations_px[keys]
        px_offset = config['crop_size'] // 2
        px_x_ = center[0] - px_offset
        px_x = center[0] + px_offset
        px_y_ = center[1] - px_offset
        px_y = center[1] + px_offset
        return image_tensor[:, :, :, px_y_:px_y, px_x_:px_x]

    @tf.function
    def processor(image_tensor, target_tensor):
        if DEBUGGING:
            pydevd.settrace(suspend=False)
        # Updated image tensor
        image_tensor_ = crop(image_tensor, list(station.keys())[0])
        image_tensor_ = tf.transpose(image_tensor_, [0, 1, 3, 4, 2])

        # Updated target tensor
        target_tensor_ = target_tensor

        return image_tensor_, target_tensor_

    return processor
