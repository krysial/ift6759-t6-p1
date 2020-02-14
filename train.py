import os
import json
import datetime
import argparse
import typing
import pandas as pd
import time
import math
import tensorflow as tf
import gc

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard,\
    ModelCheckpoint, \
    EarlyStopping, \
    CSVLogger

from models import models
from dataloader.dataloader import prepare_dataloader
from models import prepare_model


def main(
    config: typing.Dict[typing.AnyStr, typing.Any],
    admin_config_path: typing.AnyStr,
    user_config_path: typing.Optional[typing.AnyStr] = None
) -> None:
    print(config)
    print(tf.config.experimental.list_physical_devices('GPU'))

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
    ).prefetch(tf.data.experimental.AUTOTUNE)

    model = prepare_model(
        stations,
        target_time_offsets,
        config
    )

    optimizer = Adam(lr=1e-4, decay=1e-5)
    loss_fn = tf.keras.losses.MeanSquaredError()

    train_step_per_epoch = math.floor(0.9 * config.dataset_size)
    valid_step_per_epoch = math.floor(0.1 * config.dataset_size)

    writer = tf.summary.create_file_writer('results/logs/')
    tf.random.set_seed(1)

    with writer.as_default():
        for epoch in range(config.epoch):
            print('Start of epoch %d' % (epoch,))

            for step, (i_batch_train, t_batch_train) in enumerate(data_loader.take(train_step_per_epoch)):
                tf.summary.trace_on(graph=True, profiler=True)
                with tf.GradientTape() as tape:
                    targets = model(i_batch_train)
                    loss_value = loss_fn(t_batch_train, targets)

                tf.summary.scalar("train_loss", loss_value, step=step)
                tf.summary.trace_export(
                    name='profiler',
                    step=step,
                    profiler_outdir='results/logs/profiler'
                )

                writer.flush()

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                tf.keras.backend.clear_session()

                # Log every 200 batches.
                if step % math.floor(train_step_per_epoch / min(10, train_step_per_epoch)) == 0:
                    print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    print('Seen so far: %s samples' % ((step + 1) * config.batch_size))
                    tf.summary.image(
                        'training_image',
                        i_batch_train[0, :, :, :, :4],
                        step=step,
                        max_outputs=3
                    )

                del i_batch_train, t_batch_train
                gc.collect()

            for step, (i_batch_valid, t_batch_valid) in enumerate(data_loader.take(valid_step_per_epoch)):
                targets = model(i_batch_valid)
                loss_value = loss_fn(t_batch_valid, targets)

                tf.summary.scalar("valid_loss", loss_value, step=step)
                writer.flush()

                # Log every 200 batches.
                if step % math.floor(valid_step_per_epoch / min(10, valid_step_per_epoch)) == 0:
                    print('Validation loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    print('Seen so far: %s samples' % ((step + 1) * config.batch_size))

        tf.summary.trace_off()

    print(model.summary())


if __name__ == "__main__":
    DEFAULT_SEQ_LEN = 4
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
        "--dataset-size",
        type=int,
        help="dataset size",
        default=10
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
    args = parser.parse_args()

    main(
        args,
        admin_config_path=args.admin_config_path,
        user_config_path=args.user_cfg_path,
    )
