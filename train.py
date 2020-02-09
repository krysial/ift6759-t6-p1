import os
import json
import datetime
import argparse
import typing
import pandas as pd
import time
import tensorflow as tf

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard,\
    ModelCheckpoint, \
    EarlyStopping, \
    CSVLogger

from models import models
from dataloader.dataloader import prepare_dataloader
from models import prepare_model

print(tf.config.experimental.list_physical_devices('GPU'))


def main(
    config: typing.Dict[typing.AnyStr, typing.Any],
    admin_config_path: typing.AnyStr,
    user_config_path: typing.Optional[typing.AnyStr] = None
) -> None:
    user_config = {}
    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

    assert os.path.isfile(admin_config_path), f"invalid admin config file: {admin_config_path}"
    with open(admin_config_path, "r") as fd:
        admin_config = json.load(fd)

    dataframe_path = admin_config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    if "start_bound" in admin_config:
        dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(admin_config["start_bound"])]
    if "end_bound" in admin_config:
        dataframe = dataframe[dataframe.index < datetime.datetime.fromisoformat(admin_config["end_bound"])]

    target_datetimes = [datetime.datetime.fromisoformat(d) for d in admin_config["target_datetimes"]]
    assert target_datetimes and all([d in dataframe.index for d in target_datetimes])
    target_stations = admin_config["stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]
    stations = {config.station: target_stations[config.station]}

    data_loader = prepare_dataloader(
        dataframe,
        target_datetimes,
        stations,
        target_time_offsets,
        config,
        target_stations
    )

    dataset = data_loader \
        .prefetch(tf.data.experimental.AUTOTUNE) \
        .batch(config.batch_size)

    model = prepare_model(
        stations,
        target_time_offsets,
        config
    )

    optimizer = Adam(lr=1e-5, decay=1e-6)

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(
            'results',
            'checkpoints',
            config.model + '-' +
            'synthetic' if config.synthetic_data else 'real' +
            '.{epoch:03d}-{val_loss:.3f}.hdf5'
        ),
        verbose=1,
        save_best_only=True
    )

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('results', 'logs', config.model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(
        os.path.join(
            'results',
            'logs',
            config.model + '-' + 'training-' + str(timestamp) + '.log'
        )
    )

    model.fit_generator(
        dataset,
        epochs=config.epoch,
        callbacks=[tb, early_stopper, csv_logger],
        steps_per_epoch=config.dataset_size / config.batch_size
    )

    print(model.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--station",
        type=str,
        help="station to train on",
        default="BND"
    )
    parser.add_argument(
        "--synthetic-data",
        type=bool,
        help="train on synthetic mnist data",
        default=True
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        help="size of the crop frame",
        default=60
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="epoch count",
        default=10
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        help="dataset size",
        default=10000
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        help="sequence length of frames in video",
        default=6
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size of data",
        default=100
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model to be train/tested",
        default="dummy"
    )
    parser.add_argument(
        "--channels",
        dest='channels',
        help="channels to keep",
        type=str,
        nargs='*',
        default=["ch1", "ch2", "ch3", "ch4", "ch6"]
    )
    parser.add_argument(
        "-u",
        "--user_cfg_path",
        type=str,
        default=None,
        help="path to the JSON config file used to store user model/dataloader parameters"
    )
    parser.add_argument(
        "admin_config_path",
        type=str,
        help="path to the JSON config file used to store test set/evaluation parameters"
    )
    args = parser.parse_args()

    main(
        args,
        admin_config_path=args.admin_config_path,
        user_config_path=args.user_cfg_path,
    )
