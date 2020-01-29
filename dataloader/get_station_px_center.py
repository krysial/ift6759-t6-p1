

def get_station_px_center(
        dataframe: pd.DataFrame,
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],):
    """
    A function that converts the coordinate values of a point on earth to that on an image

    Args:
        dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset).
        target_stations: A dictionary of all target stations with respective coordinates and elevation 

    Returns:
        stations_px: A ``dict`` of station names as keys and their pixel coordinates as values
        L: An ``int`` value of length dimension of image
        B: An ``int`` value of bredth dimension of image
    """

    return stations_px, L, B
