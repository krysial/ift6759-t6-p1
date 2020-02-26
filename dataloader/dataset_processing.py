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
        data['images'] = crop(data['images'], list(station.keys())[0])
        data['images'] = tf.transpose(data['images'], [0, 2, 3, 1])
        return data, target_tensor

    return processor


def interpolate_GHI(data):
    """Time-based linear interpolation for missing GHI values in the given dataframe."""
    for station in ['BND', 'TBL', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF']:
        data[f'{station}_GHI'].interpolate(method='time', inplace=True)
    return data


def transposing(data, target_tensor):
    # (batch, seq, ch, dim, dim) -> (batch, seq, dim, dim, ch)
    data['images'] = tf.transpose(data['images'], [0, 1, 3, 4, 2])
    return data, target_tensor


def presaved_crop(config):
    def presaved_cropping(data, target_tensor):
        center = [40, 40]
        px_offset = config['crop_size'] // 2
        px_x_ = center[0] - px_offset
        px_x = center[0] + px_offset
        px_y_ = center[1] - px_offset
        px_y = center[1] + px_offset
        data['images'] = data['images'][:, :, px_y_:px_y, px_x_:px_x, :]
        return data, target_tensor
    return presaved_cropping


def normalize_station_GHI(data, target_tensor):
    # Quantile based normalization
    median_station_GHI = 297.487143
    quantile_diff_station_GHI = 470.059048
    target_tensor_ = tf.math.divide(tf.math.subtract(target_tensor, median_station_GHI), quantile_diff_station_GHI)
    return data, target_tensor_


def normalize_CLEARSKY_GHI(data, target_tensor):
    # Quantile based normalization
    median_CLEARSKY_GHI = 297.487143
    quantile_diff_CLEARSKY_GHI = 470.059048
    clearsky_tensor = tf.math.divide(tf.math.subtract(data['clearsky'], median_CLEARSKY_GHI), quantile_diff_CLEARSKY_GHI)
    data['clearsky'] = clearsky_tensor
    return data, target_tensor
