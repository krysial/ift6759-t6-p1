from datetime import datetime, timedelta
import typing

import numpy as np
import pandas as pd


def get_ghi_targets(
    df: pd.DataFrame,
    datetimes: typing.List[datetime],
    station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
    offsets: typing.List[timedelta],
    config: typing.Dict[typing.AnyStr, typing.Any]
    ) -> np.ndarray:
    """
    Get a station's GHI measurements for a list of datetimes and offsets.

    Args:
        df: metadata dataframe.
        datetimes: list of timestamps (datetime objects).
        station: 1-element dictionary with format {'station': (latitude, longitude, elevation)}
        offsets: list of target time offsets (timedelta objects).
        config: configuration dictionary holding any extra parameters that might be required for tuning purposes.
    Returns:
        A 2D NumPy array of GHI values, of size [#datetimes x #offsets].
    """
    # Initialize GHI target array
    targets = np.zeros((len(datetimes), len(offsets)))

    # Get station name as string
    station = list(station)[0]

    # Iterate over datetimes
    for i, dt in enumerate(datetimes):
        # Iterate over offsets
        for j, offset in enumerate(offsets):
            # Get target time with offset
            t = dt + offset
            # Get GHI value at time "t"
            try:
                targets[i, j] = df.loc[t][f'{station}_GHI']
            except KeyError:
                pass

    return targets
