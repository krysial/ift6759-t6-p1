import tensorflow as tf
import typing

MAX_GHI = 1500


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
    # Function to crop image
    def crop(image_tensor, keys):
        center = stations_px[keys]
        px_offset = config['crop_size'] // 2
        px_x_ = center[0] - px_offset
        px_x = center[0] + px_offset
        px_y_ = center[1] - px_offset
        px_y = center[1] + px_offset
        return image_tensor[:, :, px_y_:px_y, px_x_:px_x]

    def processor(data, target_tensor):
        # Updated image tensor
        image_tensor_ = crop(data['images'], list(station.keys())[0])
        image_tensor_ = tf.transpose(image_tensor_, [0, 2, 3, 1])

        # Updated target tensor
        target_tensor_ = target_tensor

        data['images'] = image_tensor_
        return data, target_tensor_

    return processor
