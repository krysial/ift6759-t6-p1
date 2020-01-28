import typing
import numpy as np


def crop_size(stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]]) -> int:
    """
    A function to find the best/smallest square crop frame size. It finds a crop size such that there are no overlaps.

    Args:
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).

    Returns:
        An ``int`` value which can be used to define the square crop frame side length.
    """

    coordinates = np.array(list(stations.values()))[:, :2]

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

    diagonal = minimum

    # return crop frame square side length
    return int(diagonal / np.sqrt(2))
