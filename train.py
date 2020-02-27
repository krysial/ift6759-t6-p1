import os
import json
import datetime
import argparse
import typing
import pandas as pd
import time
import tensorflow as tf
import numpy as np
import functools

from tensorflow.keras.callbacks import TensorBoard,\
    ModelCheckpoint, \
    EarlyStopping, \
    CSVLogger

from models import models
import dataloader.dataloader as real_prepare_dataloader
import dataloader.synthetic_dataloader as synthetic_dataloader
from models import prepare_model
from dataloader.dataset_processing import interpolate_GHI

np.random.seed(12345)
tf.random.set_seed(12345)


def main(
    config: typing.Dict[typing.AnyStr, typing.Any],
    admin_config_path: typing.AnyStr,
    user_config_path: typing.Optional[typing.AnyStr] = None
) -> None:
    print(tf.config.experimental.list_physical_devices('GPU'))

    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)
    else:
        user_config = {}
    user_config.update(vars(config))

    print(user_config)

    assert os.path.isfile(admin_config_path), f"invalid admin config file: {admin_config_path}"
    with open(admin_config_path, "r") as fd:
        admin_config = json.load(fd)

# training dataframe
    catalog_dataframe_path = user_config["train_dataset_path"]
    assert os.path.isfile(catalog_dataframe_path), f"invalid dataframe path: {catalog_dataframe_path}"
    catalog_dataframe = pd.read_pickle(catalog_dataframe_path)

    training_dataframe = catalog_dataframe.copy()
    if user_config['train_start_bound']:
        training_dataframe = \
            training_dataframe[training_dataframe.index >= datetime.datetime.fromisoformat(
                user_config['train_start_bound'])]
    if user_config["train_end_bound"]:
        training_dataframe = \
            training_dataframe[training_dataframe.index < datetime.datetime.fromisoformat(
                user_config["train_end_bound"])]
    training_dataframe = training_dataframe[training_dataframe.hdf5_16bit_path != 'nan']
    training_datetimes = training_dataframe.index.to_list()

    # val_dataframe
    val_dataframe = catalog_dataframe.copy()
    if user_config["val_start_bound"]:
        val_dataframe = \
            val_dataframe[val_dataframe.index >= datetime.datetime.fromisoformat(
                user_config["val_start_bound"])]
    if user_config["val_end_bound"]:
        val_dataframe = \
            val_dataframe[val_dataframe.index < datetime.datetime.fromisoformat(
                user_config["val_end_bound"])]
    # filtering val entries that have nan as path
    val_dataframe = val_dataframe[val_dataframe.hdf5_16bit_path != 'nan']
    validation_datetimes = val_dataframe.index.to_list()

    # Interpolate missing GHI values
    training_dataframe = interpolate_GHI(training_dataframe)
    val_dataframe = interpolate_GHI(val_dataframe)

    target_stations = admin_config["stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta(
    ) for d in admin_config["target_time_offsets"]]

    TRAIN_DT_LENGTH = functools.reduce(
        lambda agr, s: training_dataframe['{}_DAYTIME'.format(s)].sum() + agr,
        target_stations,
        0
    )

    VAL_DT_LENGTH = functools.reduce(
        lambda agr, s: val_dataframe['{}_DAYTIME'.format(s)].sum() + agr,
        target_stations,
        0
    )

    STEPS_PER_EPOCH = int(TRAIN_DT_LENGTH) // user_config["batch_size"]
    VALIDATION_STEPS = int(VAL_DT_LENGTH) // user_config["batch_size"]

    if user_config['real']:
        # real dataloader is expecting a Dict {} object in evaluation
        prepare_dataloader = real_prepare_dataloader.prepare_dataloader_train
    else:
        # load synthetic data
        prepare_dataloader = synthetic_dataloader.prepare_dataloader

    train_data_loader = prepare_dataloader(
        training_dataframe,
        training_datetimes,
        target_stations,
        target_time_offsets,
        user_config
    ).prefetch(tf.data.experimental.AUTOTUNE)

    val_data_loader = prepare_dataloader(
        val_dataframe,
        validation_datetimes,
        target_stations,
        target_time_offsets,
        user_config
    ).prefetch(tf.data.experimental.AUTOTUNE)

    timestamp = time.time()
    model_id = str(timestamp) + "_" + user_config['model']
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(
            user_config['checkpoint_path'],
            model_id + ".tf/"
        ),
        verbose=1,
        save_best_only=True,
        save_weights_only=False
    )

    tb = TensorBoard(
        log_dir=os.path.join(
            'results',
            'logs',
            model_id
        ),
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )

    early_stopper = EarlyStopping(patience=5)

    csv_logger = CSVLogger(
        os.path.join(
            'results',
            'logs',
            'backups',
            model_id + '.log'
        )
    )

    model = prepare_model(
        target_stations,
        target_time_offsets,
        user_config
    )

    model.fit_generator(
        train_data_loader,
        epochs=user_config['epoch'],
        use_multiprocessing=True,
        workers=32,
        callbacks=[tb, csv_logger, checkpointer],
        steps_per_epoch=STEPS_PER_EPOCH // 16,
        validation_steps=VALIDATION_STEPS,
        validation_data=val_data_loader
    )
    print(model.summary())


if __name__ == "__main__":
    DEFAULT_SEQ_LEN = 6
    DEFAULT_CHANNELS = ["ch1", "ch2", "ch3", "ch4", "ch6"]
    DEFAULT_IMAGE_SIZE = 80

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real",
        action='store_true',
        help="train on synthetic mnist data",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        help="size of the crop frame",
        default=DEFAULT_IMAGE_SIZE
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="epoch count",
        default=15
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        help="sequence length of frames in video",
        default=DEFAULT_SEQ_LEN
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size of data",
        default=32
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
        default=DEFAULT_CHANNELS
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
    parser.add_argument(
        "input_shape",
        help="input shape of first model layer",
        type=str,
        nargs='*',
        default=(
            DEFAULT_SEQ_LEN,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_IMAGE_SIZE,
            len(DEFAULT_CHANNELS)
        )
    )
    parser.add_argument(
        "--target_past_len",
        type=int,
        help="past number of targets to append to target output",
        default=1
    )
    parser.add_argument(
        "--target_past_interval",
        type=int,
        help="past intervel to append the target output",
        default=15
    )
    parser.add_argument(
        "--input_past_interval",
        type=int,
        help="past intervel to append the input",
        default=15
    )
    parser.add_argument(
        "--target_name",
        type=str,
        help="past target name to append",
        default="GHI"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="Learning rate for optimization",
        default=1e-5,
    )
    parser.add_argument(
        "-dr",
        "--decay_rate",
        type=float,
        help="Decay rate",
        default=1e-5,
    )
    args = parser.parse_args()

    main(
        args,
        admin_config_path=args.admin_config_path,
        user_config_path=args.user_cfg_path,
    )
