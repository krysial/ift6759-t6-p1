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

# https://arxiv.org/pdf/1709.01507.pdf


class SE_Residual_BiLRCNModel(BaseModel):

    @classmethod
    def create(
        cls,
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
    ):
        self = cls()

        # initialiser = 'lecun_normal'
        # initialiser = 'glorot_uniform'
        initialiser = config['initialiser']
        reg_lambda = config['weight_decay']
        activation = config['activation']
        dropping = config['dropping']

        def img_model():

            def squeeze_excite_residual_block(model, kernel_filters, kernel, init, reg_lambda):

                before_residual = TimeDistributed(
                    Conv2D(
                        kernel_filters, kernel, padding='same',
                        kernel_initializer=init, kernel_regularizer=l2(reg_lambda),
                    )
                )(model)
                before_residual = TimeDistributed(BatchNormalization())(before_residual)
                before_residual = Activation(activation)(before_residual)

                residual = TimeDistributed(
                    Conv2D(
                        kernel_filters, kernel, padding='same',
                        kernel_initializer=init, kernel_regularizer=l2(reg_lambda),
                    )
                )(before_residual)
                residual = TimeDistributed(BatchNormalization())(residual)
                residual = Activation(activation)(residual)

                squeeze_excite = TimeDistributed(AveragePooling2D(tuple(residual.shape[2:-1])))(residual)
                squeeze_excite = TimeDistributed(BatchNormalization())(squeeze_excite)
                squeeze_excite = TimeDistributed(Dense(kernel_filters, activation='linear', kernel_initializer=init, use_bias=False))(squeeze_excite)
                squeeze_excite = TimeDistributed(BatchNormalization())(squeeze_excite)
                squeeze_excite = Activation(activation)(squeeze_excite)
                squeeze_excite = TimeDistributed(Dense(kernel_filters // 8, activation='linear', kernel_initializer=init, use_bias=False))(squeeze_excite)
                squeeze_excite = TimeDistributed(BatchNormalization())(squeeze_excite)
                squeeze_excite = Activation(activation)(squeeze_excite)
                squeeze_excite = TimeDistributed(Dense(kernel_filters, activation='linear', kernel_initializer=init, use_bias=False))(squeeze_excite)
                squeeze_excite = TimeDistributed(BatchNormalization())(squeeze_excite)
                squeeze_excite = Activation('sigmoid')(squeeze_excite)

                after_squeeze_excite = TimeDistributed(
                    Conv2D(
                        kernel_filters, kernel, padding='same',
                        kernel_initializer=init, kernel_regularizer=l2(reg_lambda),
                    )
                )(residual)
                after_squeeze_excite = TimeDistributed(BatchNormalization())(after_squeeze_excite)
                after_squeeze_excite = Activation(activation)(after_squeeze_excite)

                scaled = Multiply()([after_squeeze_excite, squeeze_excite])

                model = Add()([before_residual, scaled])

                model = TimeDistributed(MaxPooling2D((3, 3), strides=(3, 3)))(model)
                model = TimeDistributed(Dropout(dropping))(model)
                return model

            img_input_shape = (config['seq_len'], config['crop_size'], config['crop_size'], len(config['channels']))
            img_in = Input(shape=img_input_shape)
            m = ZeroPadding3D(padding=(0, (80 - config['crop_size']) // 2, (80 - config['crop_size']) // 2), data_format=None)(img_in)
            m = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(m)
            m = squeeze_excite_residual_block(m, 64, kernel=(7, 7), init=initialiser, reg_lambda=reg_lambda)
            m = squeeze_excite_residual_block(m, 64, kernel=(5, 5), init=initialiser, reg_lambda=reg_lambda)
            m = squeeze_excite_residual_block(m, 128, kernel=(3, 3), init=initialiser, reg_lambda=reg_lambda)
            m = TimeDistributed(Flatten())(m)
            m = TimeDistributed(BatchNormalization())(m)
            m = Bidirectional(LSTM(512, dropout=dropping, return_sequences=True))(m)
            m = TimeDistributed(Dense(64, activation='linear'))(m)
            m = TimeDistributed(BatchNormalization())(m)
            m = Activation(activation)(m)
            m = Bidirectional(LSTM(256, dropout=dropping, return_sequences=False))(m)
            m = Dense(64, activation='linear')(m)
            m = BatchNormalization()(m)
            m = Activation(activation)(m)
            m = Dense(len(target_time_offsets), activation='linear')(m)
            m = BatchNormalization()(m)
            img_out = Activation(activation)(m)
            return img_in, img_out

        def clearsky_model():
            if config['real']:
                clearsky_input_shape = (4)
            else:
                clearsky_input_shape = (len(target_time_offsets))
            clearsky_in = Input(shape=clearsky_input_shape)
            clearsky_out = clearsky_in
            return clearsky_in, clearsky_out

        img_in, img_out = img_model()
        clearsky_in, clearsky_out = clearsky_model()
        o = Add()([img_out, clearsky_out])

        self.model = Model([img_in, clearsky_in], o)
        return self

    def call(self, inputs):
        return self.model([inputs['images'], inputs['clearsky']])
