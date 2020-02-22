import typing
import datetime
from models.base import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import Model
from tensorflow.keras.regularizers import *
import numpy as np
from utils.time_distributed import TimeDistributed


class BiLRCNModel(BaseModel):

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

        def img_model():

            def add_default_block(model, kernel_filters, init, reg_lambda):
                # conv
                model = TimeDistributed(
                    Conv2D(
                        kernel_filters, (3, 3), padding='same',
                        kernel_initializer=init, kernel_regularizer=l2(reg_lambda)
                    )
                )(model)
                model = TimeDistributed(BatchNormalization())(model)
                model = TimeDistributed(Activation('relu'))(model)
                # conv
                model = TimeDistributed(
                    Conv2D(
                        kernel_filters, (3, 3), padding='same',
                        kernel_initializer=init, kernel_regularizer=l2(reg_lambda)
                    )
                )(model)
                model = TimeDistributed(BatchNormalization())(model)
                model = TimeDistributed(Activation('relu'))(model)
                # max pool
                model = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(model)
                model = TimeDistributed(Dropout(0.2))(model)
                return model

            img_input_shape = (config['seq_len'], config['crop_size'], config['crop_size'], len(config['channels']))
            img_in = Input(shape=img_input_shape)
            m = TimeDistributed(
                Conv2D(
                    32, (8, 8), padding='same',
                    kernel_initializer=initialiser,
                    kernel_regularizer=l2(reg_lambda)
                )
            )(img_in)

            m = TimeDistributed(BatchNormalization())(m)
            m = TimeDistributed(Activation('relu'))(m)
            m = TimeDistributed(
                Conv2D(
                    32, (3, 3), kernel_initializer=initialiser,
                    kernel_regularizer=l2(reg_lambda)
                )
            )(m)
            m = TimeDistributed(BatchNormalization())(m)
            m = TimeDistributed(Activation('relu'))(m)
            m = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(m)
            m = TimeDistributed(Dropout(0.2))(m)
            # 2nd-5th (default) blocks
            m = add_default_block(m, 32, init=initialiser, reg_lambda=reg_lambda)
            m = add_default_block(m, 128, init=initialiser, reg_lambda=reg_lambda)
            m = add_default_block(m, 256, init=initialiser, reg_lambda=reg_lambda)
            m = add_default_block(m, 512, init=initialiser, reg_lambda=reg_lambda)
            # LSTM output head
            img_out = TimeDistributed(Flatten())(m)
            return img_in, img_out

        def clearsky_model():
            if config['real']:
                clearsky_input_shape = (config['seq_len'])
            else:
                clearsky_input_shape = (len(target_time_offsets))
            clearsky_in = Input(shape=clearsky_input_shape)
            if config['real']:
                m = Reshape((config['seq_len'], 1))(clearsky_in)
            else:
                m = RepeatVector(config['seq_len'])(clearsky_in)
            clearsky_out = Dense(512, activation='relu')(m)
            return clearsky_in, clearsky_out

        img_in, img_out = img_model()
        clearsky_in, clearsky_out = clearsky_model()

        m = Add()([img_out, clearsky_out])
        m = TimeDistributed(BatchNormalization())(m)
        m = Bidirectional(LSTM(512, dropout=0.3, return_sequences=True))(m)
        m = TimeDistributed(BatchNormalization())(m)
        m = TimeDistributed(Dense(64, activation='relu'))(m)
        m = Bidirectional(LSTM(64, dropout=0.3, return_sequences=False))(m)
        m = BatchNormalization()(m)
        m = Dense(64, activation='relu')(m)
        m = Dense(len(target_time_offsets))(m)
        m = Dense(len(target_time_offsets))(m)
        o = Dense(len(target_time_offsets))(m)

        self.model = Model([img_in, clearsky_in], o)
        return self

    def call(self, inputs):
        return self.model([inputs['images'], inputs['clearsky']])
