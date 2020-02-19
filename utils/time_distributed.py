from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.layers import Wrapper


class TimeDistributed(Wrapper):
    def __init__(self, layer, **kwargs):
        if not isinstance(layer, Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(TimeDistributed, self).__init__(layer, **kwargs)
        self.supports_masking = True

        # It is safe to use the fast, reshape-based approach with all of our
        # built-in Layers.
        self._always_use_reshape = (
            layer_utils.is_builtin_layer(layer) and
            not getattr(layer, 'stateful', False))

    def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
        """Finds non-specific dimensions in the static shapes.

        The static shapes are replaced with the corresponding dynamic shapes of the
        tensor.

        Arguments:
          init_tuple: a tuple, the first part of the output shape
          tensor: the tensor from which to get the (static and dynamic) shapes
            as the last part of the output shape
          start_idx: int, which indicate the first dimension to take from
            the static shape of the tensor
          int_shape: an alternative static shape to take as the last part
            of the output shape

        Returns:
          The new int_shape with the first part from init_tuple
          and the last part from either `int_shape` (if provided)
          or `tensor.shape`, where every `None` is replaced by
          the corresponding dimension from `tf.shape(tensor)`.
        """
        # replace all None in int_shape by K.shape
        if int_shape is None:
            int_shape = K.int_shape(tensor)[start_idx:]
        if not any(not s for s in int_shape):
            return init_tuple + tuple(int_shape)
        shape = K.shape(tensor)
        int_shape = list(int_shape)
        for i, s in enumerate(int_shape):
            if not s:
                int_shape[i] = shape[start_idx + i]
        return init_tuple + tuple(int_shape)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if len(input_shape) < 3:
            raise ValueError(
                '`TimeDistributed` Layer should be passed an `input_shape ` '
                'with at least 3 dimensions, received: ' + str(input_shape))
        # Don't enforce the batch or time dimension.
        self.input_spec = InputSpec(shape=[None, None] + input_shape[2:])
        child_input_shape = [input_shape[0]] + input_shape[2:]
        super(TimeDistributed, self).build(tuple(child_input_shape))
        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        child_input_shape = tensor_shape.TensorShape([input_shape[0]] +
                                                     input_shape[2:])
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        if not isinstance(child_output_shape, tensor_shape.TensorShape):
            child_output_shape = tensor_shape.TensorShape(child_output_shape)
        child_output_shape = child_output_shape.as_list()
        timesteps = input_shape[1]
        return tensor_shape.TensorShape([child_output_shape[0], timesteps] +
                                        child_output_shape[1:])

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        if generic_utils.has_arg(self.layer.call, 'training'):
            kwargs['training'] = training

        input_shape = K.int_shape(inputs)
        if input_shape[0] and not self._always_use_reshape:
            # batch size matters, use rnn-based implementation
            def step(x, _):
                output = self.layer(x, **kwargs)
                return output, []

            _, outputs, _ = K.rnn(
                step,
                inputs,
                initial_states=[],
                input_length=input_shape[1],
                unroll=False)
            y = outputs
        else:
            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.
            input_length = input_shape[1]
            if not input_length:
                input_length = array_ops.shape(inputs)[1]
            inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
            # Shape: (num_samples * timesteps, ...). And track the
            # transformation in self._input_map.
            inputs = array_ops.reshape(inputs, inner_input_shape)
            # (num_samples * timesteps, ...)
            if generic_utils.has_arg(self.layer.call, 'mask') and mask is not None:
                inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
                kwargs['mask'] = K.reshape(mask, inner_mask_shape)
            y = self.layer(inputs, **kwargs)
            # Shape: (num_samples, timesteps, ...)
            output_shape = self.compute_output_shape(input_shape).as_list()
            output_shape = self._get_shape_tuple(
                (-1, input_length), y, 1, output_shape[2:])
            y = array_ops.reshape(y, output_shape)

        return y

    def compute_mask(self, inputs, mask=None):
        # cases need to call the layer.compute_mask when input_mask is None:
        # Masking layer and Embedding layer with mask_zero
        input_shape = K.int_shape(inputs)
        if input_shape[0]:
            # batch size matters, we currently do not handle mask explicitly
            return mask
        inner_mask = mask
        if inner_mask is not None:
            inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
            inner_mask = K.reshape(inner_mask, inner_mask_shape)
        inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
        inner_inputs = array_ops.reshape(inputs, inner_input_shape)
        output_mask = self.layer.compute_mask(inner_inputs, inner_mask)
        if output_mask is None:
            if mask is None:
                return None
            # input_mask is not None, and output_mask is None:
            # we should return a not-None mask
            output_mask = mask
            for _ in range(2, len(K.int_shape(mask))):
                output_mask = K.any(output_mask, axis=-1)
        else:
            # output_mask is not None. We need to reshape it
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(inputs)[1]
            output_mask_int_shape = K.int_shape(output_mask)
            if output_mask_int_shape is None:
                # if the output_mask does not have a static shape,
                # its shape must be the same as mask's
                if mask is not None:
                    output_mask_int_shape = K.int_shape(mask)
                else:
                    output_mask_int_shape = K.compute_output_shape(input_shape)[
                        :-1]
            output_mask_shape = self._get_shape_tuple(
                (-1, input_length), output_mask, 1, output_mask_int_shape[1:])
            output_mask = K.reshape(output_mask, output_mask_shape)
        return output_mask
