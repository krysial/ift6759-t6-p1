from get_station_px_center import get_station_px_center
from crop_size import crop_size


def get_crop_frame_size(
        dataframe: pd.DataFrame,
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],) -> int:
    """
    A function to return an ``int`` value corresponding to the crop frame side length
    """

    stations_px, L, B = get_station_px_center(dataframe, target_stations)
    crop_frame_size = crop_size(stations_px, L, B)
    return crop_frame_size
