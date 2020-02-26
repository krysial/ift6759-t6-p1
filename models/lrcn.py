import typing
import datetime
from models.base import BaseModel
import tensorflow as tf

from utils.time_distributed import TimeDistributed

from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Concatenate, RepeatVector
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

        self.conv2DPipe = Sequential()
        self.conv2DPipe.add(
            TimeDistributed(
                Conv2D(
                    32, (8, 8), padding='same',
                    kernel_initializer=initialiser,
                    kernel_regularizer=l2(reg_lambda)
                ),
                input_shape=(config['seq_len'], config['crop_size'], config['crop_size'], len(config['channels']))
            )
        )
        self.conv2DPipe.add(TimeDistributed(BatchNormalization()))
        self.conv2DPipe.add(TimeDistributed(Activation('relu')))
        self.conv2DPipe.add(
            TimeDistributed(
                Conv2D(
                    32, (3, 3), kernel_initializer=initialiser,
                    kernel_regularizer=l2(reg_lambda)
                )
            )
        )
        self.conv2DPipe.add(TimeDistributed(BatchNormalization()))
        self.conv2DPipe.add(TimeDistributed(Activation('relu')))
        self.conv2DPipe.add(
            TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2))
            )
        )

        # 2nd-5th (default) blocks
        self.conv2DPipe = self.add_default_block(self.conv2DPipe, 32, init=initialiser, reg_lambda=reg_lambda)
        self.conv2DPipe = self.add_default_block(self.conv2DPipe, 128, init=initialiser, reg_lambda=reg_lambda)
        self.conv2DPipe = self.add_default_block(self.conv2DPipe, 256, init=initialiser, reg_lambda=reg_lambda)
        self.conv2DPipe = self.add_default_block(self.conv2DPipe, 512, init=initialiser, reg_lambda=reg_lambda)

        # LSTM output head
        self.conv2DPipe.add(TimeDistributed(Flatten()))
        self.extraFeatures = Sequential()
        self.extraFeatures.add(RepeatVector(config['seq_len']))
        self.extraFeatures.add(Dense(1012))

        self.sequence = Sequential()
        self.sequence.add(Concatenate(axis=-1))
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

    def call(self, inputs, training=None):
        images = self.conv2DPipe(inputs['images'])
        clearsky = self.extraFeatures(inputs['clearsky'])

        return self.sequence([images, clearsky])
