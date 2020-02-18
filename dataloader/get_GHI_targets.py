from datetime import datetime, timedelta
import typing

import numpy as np
import pandas as pd


def get_GHI_targets(
        df: pd.DataFrame,
        datetimes: typing.List[datetime],
        station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        offsets: typing.List[timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]) -> np.ndarray:
    """
    Get a station's GHI measurements for a list of datetimes and offsets.

    Args:
        df: metadata dataframe.
        datetimes: list of timestamps (datetime objects) to provide targets for.
        station: 1-element dictionary with format {'station': (latitude, longitude, elevation)}
        offsets: list of target time offsets (timedelta objects) (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required for tuning purposes.
    Returns:
        A 2D NumPy array of GHI values, of size [#datetimes, #offsets + config['target_past_len'] - 1].
    """
    # Add past images (if any) as offsets
    for i in range(config['target_past_len'] - 1):
        offsets.insert(0, timedelta(minutes=-(i + 1) * 15))

    # Initialize target array
    targets = np.zeros((len(datetimes), len(offsets)))

    # Get station name as string
    station = list(station)[0]

    # Iterate over datetimes
    for i, dt in enumerate(datetimes):
        # Iterate over offsets
        for j, offset in enumerate(offsets):
            # Get target time with offset
            t = dt + offset
            # Get target value at time "t"
            # If target value is an invalid type or does not exist, pass
            try:
                ghi = df.loc[t][f"{station}_{config['target_name']}"]
                assert isinstance(ghi, np.float64)
            except (AssertionError, KeyError):
                pass
            else:
                targets[i, j] = ghi

    return targets


# Example usage
if __name__ == '__main__':
    # Get metadata dataframe
    df = pd.read_pickle('/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl')

    # Convert list of datetime strings into datetime objects
    datetimes = ["2015-01-01T12:45:00", "2015-01-01T19:00:00"]
    datetimes = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in datetimes]

    # Dictionary with one entry for station of interest
    station = {'DRA': (0, 0, 0)}

    # Convert list of offset strings into timedelta objects
    offsets = ["P0DT0H0M0S", "P0DT1H0M0S", "P0DT3H0M0S", "P0DT6H0M0S"]
    offsets = [timedelta(hours=int(x[4])) for x in offsets]

    # Configuration dictionary
    config = {'target_name': 'GHI', 'target_past_len': 1}

    # Call function and print results
    targets = get_GHI_targets(df, datetimes, station, offsets, config)
    print(targets)
