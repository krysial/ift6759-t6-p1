

def crop_size(
        stations_px: typing.Dict[typing.AnyStr, typing.Tuple[float, float]],
        L: int, B: int,) -> int:
    """
    A function to find the best/smallest square crop frame size. It finds a crop size such that
    there are no overlaps and is a valid crop for all stations.

    Args:
        stations: a map of station names with their pixel coordinates on image.
        L: length dimension of image
        B: bredth dimension of image

    Returns:
        An ``int`` value which can be used to define the square crop frame side length.
    """

    return sq_crop_side_len
