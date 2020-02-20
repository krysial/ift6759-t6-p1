import typing
import datetime
from models.base import BaseModel
import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

import numpy as np


class ConvLSTMModel(BaseModel):
    @classmethod
    def create(
        cls,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
    ):
        self = cls()

        img_input_shape = (config['seq_len'], config['crop_size'], config['crop_size'], len(config['channels']))

        i = Input(shape=img_input_shape)

        m = BatchNormalization()(i)
        m = ConvLSTM2D(filters=64,
                       kernel_size=(3, 3),
                       activation='relu',
                       padding='valid', return_sequences=True)(m)

        for _ in range(config['seq_len'] - 4):
            m = BatchNormalization()(m)
            m = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(m)
            m = Dropout(0.33)(m)
            m = ConvLSTM2D(filters=64 * 2**(_ // 2),
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same', return_sequences=True)(m)

        m = BatchNormalization()(m)
        m = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(m)
        m = Dropout(0.33)(m)
        m = ConvLSTM2D(filters=64 * 2**((_ + 1)),
                       kernel_size=(3, 3),
                       activation='relu',
                       padding='same', return_sequences=True)(m)

        m = BatchNormalization()(m)
        m = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(m)
        m = Dropout(0.33)(m)
        m = ConvLSTM2D(filters=64 * 2**((_ + 2)),
                       kernel_size=(3, 3),
                       activation='relu',
                       padding='same', return_sequences=True)(m)

        m = TimeDistributed(Flatten(), name='TD.Flatten')(m)
        m = TimeDistributed(BatchNormalization(), name='TD.BN1')(m)
        m = TimeDistributed(Dropout(0.2), name='TD.Dropout1')(m)
        m = TimeDistributed(Dense(1024, activation='relu'), name='TD.Dense1')(m)
        m = TimeDistributed(BatchNormalization(), name='TD.BN2')(m)
        m = TimeDistributed(Dropout(0.2), name='TD.Dropout2')(m)
        m = TimeDistributed(Dense(1024, activation='relu'), name='TD.Dense2')(m)
        m = TimeDistributed(BatchNormalization(), name='TD.BN3')(m)
        m = TimeDistributed(Dropout(0.2), name='TD.Dropout3')(m)
        m = Flatten()(m)

        m = Dense(4)(m)
        m = Dense(4)(m)
        m = Dense(4)(m)
        o = Dense(4)(m)

        self.model = Model(i, o)
        return self

    def call(self, inputs):
        return self.model(inputs['images'])
