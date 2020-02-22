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


class Conv3DModel(BaseModel):
    @classmethod
    def create(
        cls,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
    ):
        self = cls()

        def convblock1(model, filters, kernel, activation, conv_stride, padding, pool_size, pool_stride, drop_prob):
            model = Conv3D(filters, kernel, activation=activation, strides=conv_stride, padding=padding)(model)
            model = TimeDistributed(BatchNormalization())(model)
            model = MaxPooling3D(pool_size=pool_size, strides=pool_stride)(model)
            model = TimeDistributed(Dropout(drop_prob))(model)
            return model

        def convblock2(model, filters, kernel, activation, conv_stride, padding, pool_size, pool_stride, drop_prob):
            model = Conv3D(filters, kernel, activation=activation, strides=conv_stride, padding=padding)(model)
            model = TimeDistributed(BatchNormalization())(model)
            model = convblock1(model, filters, kernel, activation, conv_stride, padding, pool_size, pool_stride, drop_prob)
            return model

        def denseblock(model, drop_prob, units, activation):
            model = BatchNormalization()(model)
            model = Dropout(drop_prob)(model)
            model = Dense(units, activation=activation)(model)
            return model

        def img_model():
            img_input_shape = (config['seq_len'], config['crop_size'], config['crop_size'], len(config['channels']))
            img_in = Input(shape=img_input_shape)
            m = TimeDistributed(BatchNormalization())(img_in)
            m = convblock2(
                model=m,
                filters=64,
                kernel=(5, 5, 5),
                activation='relu',
                conv_stride=(1, 1, 1),
                padding='same',
                pool_size=(1, 2, 2),
                pool_stride=(1, 1, 1),
                drop_prob=0.2,
            )
            m = convblock2(
                model=m,
                filters=64,
                kernel=(3, 3, 3),
                activation='relu',
                conv_stride=(1, 1, 1),
                padding='same',
                pool_size=(1, 2, 2),
                pool_stride=(1, 1, 1),
                drop_prob=0.2,
            )
            m = convblock2(
                model=m,
                filters=128,
                kernel=(2, 2, 2),
                activation='relu',
                conv_stride=(1, 1, 1),
                padding='valid',
                pool_size=(1, 3, 3),
                pool_stride=(1, 2, 2),
                drop_prob=0.2,
            )
            m = convblock2(
                model=m,
                filters=128,
                kernel=(2, 2, 2),
                activation='relu',
                conv_stride=(1, 1, 1),
                padding='valid',
                pool_size=(1, 3, 3),
                pool_stride=(1, 2, 2),
                drop_prob=0.2,
            )
            m = convblock1(
                model=m,
                filters=512,
                kernel=(1, 3, 3),
                activation='relu',
                conv_stride=(1, 1, 1),
                padding='valid',
                pool_size=(2, 3, 3),
                pool_stride=(1, 2, 2),
                drop_prob=0.2,
            )
            img_out = Flatten()(m)
            return img_in, img_out

        def clearsky_model():
            if config['real']:
                clearsky_input_shape = (config['seq_len'])
            else:
                clearsky_input_shape = (len(target_time_offsets))
            clearsky_in = Input(shape=clearsky_input_shape)
            clearsky_out = Dense(2048, activation='relu')(clearsky_in)
            return clearsky_in, clearsky_out

        img_in, img_out = img_model()
        clearsky_in, clearsky_out = clearsky_model()

        m = Add()([img_out, clearsky_out])
        m = denseblock(
            model=m,
            drop_prob=0.3,
            units=1024,
            activation='relu',
        )
        m = denseblock(
            model=m,
            drop_prob=0.3,
            units=1024,
            activation='relu',
        )
        m = denseblock(
            model=m,
            drop_prob=0.3,
            units=4,
            activation='linear',
        )
        o = denseblock(
            model=m,
            drop_prob=0.3,
            units=len(target_time_offsets),
            activation='linear',
        )

        self.model = Model([img_in, clearsky_in], o)
        return self

    def call(self, inputs):
        return self.model([inputs['images'], inputs['clearsky']])
