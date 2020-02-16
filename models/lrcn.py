import typing
import datetime
from models.base import BaseModel
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import (
    Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, BatchNormalization, Activation
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
        reg_lambda = 0.001

        self.sequence = Sequential()
        self.sequence.add(
            TimeDistributed(
                Conv2D(
                    32, (8, 8), padding='same',
                    kernel_initializer=initialiser,
                    kernel_regularizer=l2(reg_lambda)
                )
            )
        )
        self.sequence.add(TimeDistributed(BatchNormalization()))
        self.sequence.add(TimeDistributed(Activation('relu')))
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
        self.sequence = self.add_default_block(self.sequence, 256, init=initialiser, reg_lambda=reg_lambda)
        self.sequence = self.add_default_block(self.sequence, 512, init=initialiser, reg_lambda=reg_lambda)

        # LSTM output head
        self.sequence.add(TimeDistributed(Flatten()))
        self.sequence.add(Dense(512))
        self.sequence.add(Dense(512))
        self.sequence.add(LSTM(512, return_sequences=True, dropout=0.3))
        self.sequence.add(LSTM(512, dropout=0.3))
        self.sequence.add(Dense(512))
        self.sequence.add(Dense(len(target_time_offsets)))

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
        return self.sequence(inputs)
