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
import dataloader.dataloader as real_prepare_dataloader
import dataloader.synthetic_dataloader as synthetic_dataloader
from models import prepare_model
from dataloader.dataset_processing import dataset_concat_seq_images


def main(
    config: typing.Dict[typing.AnyStr, typing.Any],
    admin_config_path: typing.AnyStr,
    user_config_path: typing.Optional[typing.AnyStr] = None
) -> None:
    print(config)
    print(tf.config.experimental.list_physical_devices('GPU'))

    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)
    else:
        user_config = {}
    user_config.update(vars(config))

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
    stations = {user_config['station']: target_stations[user_config['station']]}

    DATASET_LENGTH = len(dataframe)
    STEPS_PER_EPOCH = int(0.9 * DATASET_LENGTH)
    VALIDATION_STEPS = int(0.1 * DATASET_LENGTH)

    user_config['seq_len'] = 1  # TODO: this needs to be fixed

    if user_config['real']:
        # real dataloader is expecting a Dict {} object in evaluation
        prepare_dataloader = real_prepare_dataloader.prepare_dataloader
    else:
        # load synthetic data
        prepare_dataloader = synthetic_dataloader.prepare_dataloader

    data_loader = prepare_dataloader(
        dataframe,
        target_datetimes,
        stations,
        target_time_offsets,
        user_config
    ).prefetch(tf.data.experimental.AUTOTUNE)

    if user_config['stack_seqs']:
        data_loader = data_loader.map(dataset_concat_seq_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    model = prepare_model(
        stations,
        target_time_offsets,
        user_config
    )

    optimizer = Adam(lr=1e-10, decay=1e-12)

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(
            'results',
            'checkpoints',
            user_config['model'] + '-' +
            'synthetic' if not user_config['real'] else 'real' +
            '.{epoch:03d}-{val_loss:.3f}.hdf5'
        ),
        verbose=1,
        save_best_only=True
    )

    # class MyCustomCallback(tf.keras.callbacks.Callback):
    #     def on_train_batch_begin(self, batch, logs=None):
    #         print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    #     def on_train_batch_end(self, batch, logs=None):
    #         print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    #     def on_test_batch_begin(self, batch, logs=None):
    #         print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    #     def on_test_batch_end(self, batch, logs=None):
    #         print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    # Helper: TensorBoard
    tb = TensorBoard(
        log_dir=os.path.join('results', 'logs', user_config['model']),
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(
        os.path.join(
            'results',
            'logs',
            user_config['model'] + '-' + 'training-' + str(timestamp) + '.log'
        )
    )

    model.fit(
        data_loader,
        epochs=user_config['epoch'],
        use_multiprocessing=True,
        workers=32,
        callbacks=[tb, csv_logger, early_stopper],
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=data_loader,
        validation_steps=VALIDATION_STEPS
    )

    print(model.summary())


if __name__ == "__main__":
    DEFAULT_SEQ_LEN = 6
    DEFAULT_CHANNELS = ["ch1", "ch2", "ch3", "ch4", "ch6"]
    DEFAULT_IMAGE_SIZE = 40

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--station",
        type=str,
        help="station to train on",
        default="BND"
    )
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
        "--stack_seqs",
        action='store_true',
        help="stack seq images as channels in output tensor",
        default=False
    )
    args = parser.parse_args()

    main(
        args,
        admin_config_path=args.admin_config_path,
        user_config_path=args.user_cfg_path,
    )
