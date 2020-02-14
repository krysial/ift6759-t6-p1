import pandas as pd
import typing
import datetime

from dataloader.get_GHI_targets import get_GHI_targets
from dataloader.get_raw_images import get_raw_images


try:
    import pydevd
    DEBUGGING = True
except ImportError:
    DEBUGGING = False


def create_data_generator(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        station: typing.Dict[
            typing.AnyStr,
            typing.Tuple[float, float, float]
        ],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
):
    """
    A function to create a generator to yield data to the dataloader
    """
    def create_generator():
        if DEBUGGING:
            pydevd.settrace(suspend=False)

        for i in range(0, len(target_datetimes), config['batch_size']):
            if config['train']:
                batch_of_datetimes = list(dataframe.index.to_numpy(dtype=str))[i:i + config['batch_size']]
            else:
                batch_of_datetimes = target_datetimes[i:i + config['batch_size']]
            targets = get_GHI_targets(
                dataframe,
                batch_of_datetimes,
                station,
                target_time_offsets,
                config,
            )
            images = get_raw_images(dataframe, batch_of_datetimes, config)
            yield images, targets

    return create_generator
