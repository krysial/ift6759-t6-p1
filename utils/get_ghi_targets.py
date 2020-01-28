from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def get_ghi_targets(df, datetimes, station, offsets=[0, 1, 3, 6]):
    """
    Get a station's GHI measurements for a list of datetimes and offsets.

    Args:
        df: metadata dataframe.
        datetimes: list of timestamps (datetime objects).
        station: station name as a string (e.g. "BND").
        offsets: list of integers corresponding to hour offsets; default [0,1,3,6].
    Returns:
        A 2D NumPy array of GHI values, of size [#datetimes x #offsets].
    """
    # Initialize GHI target array
    targets = np.zeros((len(datetimes), len(offsets)))

    # Iterate over datetimes
    for i, dt in enumerate(datetimes):
        # Iterate over offsets
        for j, offset in enumerate(offsets):
            # Get target time
            t = dt + timedelta(hours=offset)
            # Get GHI value at time "t"
            try:
                targets[i, j] = df.loc[t][f'{station}_GHI']
            except KeyError:
                pass

    return targets
