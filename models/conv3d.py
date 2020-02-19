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


class Conv3DModel(BaseModel):
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
        self.sequece.add(Conv3D(64, (2, 2, 2), activation='relu', strides=(1, 1, 1)))

        for _ in range(config['seq_len'] - 2):
            self.sequece.add(BatchNormalization())
            self.sequece.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1)))
            self.sequece.add(Dropout(0.33))
            self.sequece.add(Conv3D(64 * 2**(_ // 2), (2, 2, 2), activation='relu', strides=(1, 1, 1)))

        self.sequece.add(BatchNormalization())
        self.sequece.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))
        self.sequece.add(Dropout(0.33))
        self.sequece.add(Conv3D(64 * 2**((_ + 1)), (1, 3, 3), activation='relu', strides=(1, 1, 1)))
        self.sequece.add(BatchNormalization())
        self.sequece.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))
        self.sequece.add(Dropout(0.33))
        self.sequece.add(Conv3D(64 * 2**((_ + 2)), (1, 3, 3), activation='relu', strides=(1, 1, 1)))

        self.sequece.add(BatchNormalization())
        self.sequece.add(Flatten())
        self.sequece.add(BatchNormalization())
        self.sequece.add(Dropout(0.2))
        self.sequece.add(Dense(1024))
        self.sequece.add(BatchNormalization())
        self.sequece.add(Dropout(0.2))
        self.sequece.add(Dense(1024))
        self.sequece.add(BatchNormalization())
        self.sequece.add(Dropout(0.2))

        self.sequece.add(Dense(4))

        return self

    def call(self, inputs):
        return self.sequece(inputs['images'])
