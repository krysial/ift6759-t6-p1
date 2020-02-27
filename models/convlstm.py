import typing
import datetime
from models.base import BaseModel
import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

from utils.time_distributed import TimeDistributed

import numpy as np


class ConvLSTMModel(BaseModel):
    def ConvLSTM_block(self, model, filters, kernel_size, activation, padding, return_sequences, pool_size, pool_strides, drop_prob):
        model = ConvLSTM2D(filters=filters,
                           kernel_size=kernel_size,
                           padding=padding,
                           return_sequences=return_sequences)(model)
        model = BatchNormalization()(model)
        model = Activation(activation)(model)
        model = MaxPooling3D(pool_size=pool_size, strides=pool_strides)(model)
        model = Dropout(drop_prob)(model)
        return model

    def TD_Dense_block(self, model, drop_prob, units, activation, name):
        model = TimeDistributed(Dense(units, activation=activation), name='TD.Dense' + name)(model)
        model = TimeDistributed(BatchNormalization(), name='TD.BN' + name)(model)
        model = TimeDistributed(Dropout(drop_prob), name='TD.Dropout' + name)(model)
        return model

    @classmethod
    def create(
        cls,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
    ):
        self = cls()
        activation = 'relu'

        def img_model():
            img_input_shape = (config['seq_len'], config['crop_size'], config['crop_size'], len(config['channels']))
            img = Input(shape=img_input_shape)
            m = ZeroPadding3D(padding=(0, (80 - config['crop_size']) // 2, (80 - config['crop_size']) // 2), data_format=None)(img)
            m = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(m)
            m = self.ConvLSTM_block(
                model=m,
                filters=32,
                kernel_size=(5, 5),
                activation=activation,
                padding='same',
                return_sequences=True,
                pool_size=(1, 2, 2),
                pool_strides=(1, 2, 2),
                drop_prob=0.33,
            )
            m = self.ConvLSTM_block(
                model=m,
                filters=64,
                kernel_size=(3, 3),
                activation=activation,
                padding='valid',
                return_sequences=True,
                pool_size=(1, 3, 3),
                pool_strides=(1, 2, 2),
                drop_prob=0.33,
            )
            m = TimeDistributed(Flatten(), name='TD.Flatten')(m)
            return img, m

        def clearsky_model():
            if config['real']:
                clearsky_input_shape = (config['seq_len'])
            else:
                clearsky_input_shape = (len(target_time_offsets))
            clearsky = Input(shape=clearsky_input_shape)
            if config['real']:
                m = Reshape((config['seq_len'], 1))(clearsky)
            else:
                m = RepeatVector(config['seq_len'])(clearsky)
            m = Dense(4096, activation=activation)(m)
            return clearsky, m

        img_in, img_out = img_model()
        clearsky_in, clearsky_out = clearsky_model()

        m = Add()([img_out, clearsky_out])

        m = self.TD_Dense_block(
            model=m,
            drop_prob=0.2,
            units=1024,
            activation=activation,
            name='1'
        )

        m = self.TD_Dense_block(
            model=m,
            drop_prob=0.2,
            units=512,
            activation=activation,
            name='2'
        )

        m = self.TD_Dense_block(
            model=m,
            drop_prob=0.2,
            units=128,
            activation=activation,
            name='3'
        )

        m = LSTM(64, return_sequences=True)(m)
        m = LSTM(64, return_sequences=True)(m)

        m = Flatten()(m)
        m = Dense(len(target_time_offsets))(m)
        m = Dense(len(target_time_offsets))(m)
        o = Dense(len(target_time_offsets))(m)

        self.model = Model([img_in, clearsky_in], o)
        return self

    def call(self, inputs, training=None):
        return self.model([inputs['images'], inputs['clearsky']])
