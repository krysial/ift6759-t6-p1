import pandas as pd
import typing
from datetime import datetime, timedelta
import numpy as np

from copy import deepcopy as copy
from dataloader.get_GHI_targets import get_GHI_targets


def get_column_from_dataframe(
        df: pd.DataFrame,
        datetimes: typing.List[datetime],
        station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        offsets: typing.List[timedelta],
        column: typing.AnyStr,
        config: typing.Dict[typing.AnyStr, typing.Any]) -> np.ndarray:
    """
    Get a station's GHI measurements for a list of datetimes and offsets.

    Args:
        df: metadata dataframe.
        datetimes: list of timestamps (datetime objects) to provide targets for.
        station: 1-element dictionary with format {'station': (latitude, longitude, elevation)}
        offsets: list of target time offsets (timedelta objects) (by definition: [T=0, T+1h, T+3h, T+6h]).
        column: name of column to return with seq_len past sequences.
        config: configuration dictionary holding any extra parameters that might be required for tuning purposes.
    Returns:
        A 2D NumPy array of GHI values, of size [#datetimes, #offsets + config['target_past_len'] - 1].
    """
    new_config = copy(config)
    new_config['target_name'] = column
    new_config['target_past_len'] = config['seq_len']
    new_config['target_past_interval'] = config['input_past_interval']
    new_offset = [copy(offsets)[0]]
    COLUMN = get_GHI_targets(
        df,
        datetimes,
        station,
        new_offset,
        new_config
    )
    return COLUMN
