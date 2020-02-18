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

        self.sequece = Sequential()
        self.sequece.add(BatchNormalization())
        self.sequece.add(ConvLSTM2D(filters=64, 
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='valid', return_sequences=True))

        for _ in range(config.seq_len - 4):
            self.sequece.add(BatchNormalization())
            self.sequece.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
            self.sequece.add(Dropout(0.33))
            self.sequece.add(ConvLSTM2D(filters=64*2**(_//2), 
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same', return_sequences=True))

        self.sequece.add(BatchNormalization())
        self.sequece.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        self.sequece.add(Dropout(0.33))
        self.sequece.add(ConvLSTM2D(filters=64*2**((_+1)), 
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same', return_sequences=True))

        self.sequece.add(BatchNormalization())
        self.sequece.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))
        self.sequece.add(Dropout(0.33))
        self.sequece.add(ConvLSTM2D(filters=64*2**((_+2)), 
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='same', return_sequences=True))

        self.sequece.add(TimeDistributed(Flatten(), name='TD.Flatten'))
        self.sequece.add(TimeDistributed(BatchNormalization(), name='TD.BN1'))
        self.sequece.add(TimeDistributed(Dropout(0.2), name='TD.Dropout1'))
        self.sequece.add(TimeDistributed(Dense(1024, activation='relu'), name='TD.Dense1'))
        self.sequece.add(TimeDistributed(BatchNormalization(), name='TD.BN2'))
        self.sequece.add(TimeDistributed(Dropout(0.2), name='TD.Dropout2'))
        self.sequece.add(TimeDistributed(Dense(1024, activation='relu'), name='TD.Dense2'))
        self.sequece.add(TimeDistributed(BatchNormalization(), name='TD.BN3'))
        self.sequece.add(TimeDistributed(Dropout(0.2), name='TD.Dropout3'))
        self.sequece.add(Flatten())

        self.sequece.add(Dense(4))
        self.sequece.add(Dense(4))
        self.sequece.add(Dense(4))
        self.sequece.add(Dense(4))

        return self

    def call(self, inputs):
        return self.sequece(inputs)
