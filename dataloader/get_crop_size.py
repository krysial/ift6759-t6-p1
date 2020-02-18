import typing
import numpy as np
import tensorflow as tf


def get_crop_size(
        stations_px: typing.Dict[typing.AnyStr, typing.Tuple[int, int]],
        data_loader: tf.data.Dataset,) -> int:
    """
    A function to find the best/smallest square crop frame size. It finds a crop size such that
    there are no overlaps and is a valid crop for all stations.
    Args:
        stations: a map of station names with their pixel coordinates on image.
        data_loader: the ``tf.data.Dataset`` holding raw images and targets.
    Returns:
        An ``int`` value which can be used to define the square crop frame side length.
    """

    def get_L_B(data_loader: tf.data.Dataset,) -> typing.Tuple[int, int]:
        """
        A util function to extract and return image dimensions from data_loader
        """

        for data, target in data_loader:
            break
        L, B = data['images'].shape[-2:]
        return L, B

    # station coordinates and image dimensions
    coordinates = np.array(list(stations_px.values()))[:, :2]
    L, B = get_L_B(data_loader)

    # coordinates of edge points nearest to the stations along the edges
    l_min = [0, coordinates[coordinates[:, 0].argmin()][1]]
    b_min = [coordinates[coordinates[:, 1].argmin()][0], 0]
    l_max = [L - 1, coordinates[coordinates[:, 0].argmin()][1]]
    b_max = [coordinates[coordinates[:, 1].argmin()][0], B - 1]

    # updating coordinates
    coordinates = np.append(coordinates, [l_min, b_min, l_max, b_max], axis=0)

    # function to calculate distance between 2 points in n-Dimenstions
    def calcul_distance(coordinate1: np.array, coordinate2: np.array) -> float:
        return np.sqrt(np.sum(np.square(coordinate2 - coordinate1)))

    # finding minimum distance (or) crop frame diagonal length between set of points
    minimum = np.infty
    for c1 in coordinates:
        for c2 in coordinates:
            distance = calcul_distance(c1, c2)
            if distance < minimum and distance != 0:
                minimum = distance

    # setting diagonal equal to the minimum link distance
    diagonal = minimum

    # return crop frame square side length
    sq_crop_side_len = int(diagonal / np.sqrt(2))
    return sq_crop_side_len
