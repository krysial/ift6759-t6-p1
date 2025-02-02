import typing
import datetime
from models.base import BaseModel
import tensorflow as tf

from utils.time_distributed import TimeDistributed

from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Add, Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Concatenate, RepeatVector
from tensorflow.keras.layers import (
    Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, BatchNormalization, Activation, Dropout
)
from tensorflow.keras.regularizers import l2

import numpy as np


class LRCNModel(BaseModel):
    @classmethod
    def create(
        cls,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
    ):
        self = cls()

        initialiser = 'glorot_uniform'
        reg_lambda = 0

        self.sequence = Sequential()
        self.sequence.add(
            TimeDistributed(
                Conv2D(
                    32, (8, 8), padding='same',
                    kernel_initializer=initialiser,
                    kernel_regularizer=l2(reg_lambda)
                ),
                input_shape=(config['seq_len'], config['crop_size'], config['crop_size'], len(config['channels']))
            )
        )
        self.sequence.add(TimeDistributed(BatchNormalization()))
        self.sequence.add(TimeDistributed(Activation('relu')))
        self.sequence.add(TimeDistributed(Dropout(0.5)))
        self.sequence.add(
            TimeDistributed(
                Conv2D(
                    32, (3, 3), kernel_initializer=initialiser,
                    kernel_regularizer=l2(reg_lambda)
                )
            )
        )
        self.sequence.add(TimeDistributed(BatchNormalization()))
        self.sequence.add(TimeDistributed(Activation('relu')))
        self.sequence.add(
            TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2))
            )
        )

        # 2nd-5th (default) blocks
        self.sequence = self.add_default_block(self.sequence, 32, init=initialiser, reg_lambda=reg_lambda)
        self.sequence = self.add_default_block(self.sequence, 128, init=initialiser, reg_lambda=reg_lambda)
        # self.sequence = self.add_default_block(self.sequence, 256, init=initialiser, reg_lambda=reg_lambda)
        # self.sequence = self.add_default_block(self.sequence, 512, init=initialiser, reg_lambda=reg_lambda)

        # LSTM output head
        self.sequence.add(Flatten())
        self.sequence.add(Dense(512, activation='tanh'))
        self.sequence.add(Dense(512, activation='tanh'))
        self.sequence.add(Dropout(0.5))
        self.sequence.add(Dense(len(target_time_offsets)))

        self.last = Add()

        return self

    def add_default_block(self, model, kernel_filters, init, reg_lambda):
        # conv
        model.add(
            TimeDistributed(
                Conv2D(
                    kernel_filters, (3, 3), padding='same',
                    kernel_initializer=init, kernel_regularizer=l2(reg_lambda)
                )
            )
        )
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        # conv
        model.add(
            TimeDistributed(
                Conv2D(
                    kernel_filters, (3, 3), padding='same',
                    kernel_initializer=init, kernel_regularizer=l2(reg_lambda)
                )
            )
        )
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        # max pool
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        return model

    def call(self, inputs):
        images = self.sequence(inputs['images'])
        clearsky = inputs['clearsky']

        return self.last([images, clearsky])
