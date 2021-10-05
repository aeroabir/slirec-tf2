import tensorflow as tf
from tensorflow import keras
import logging
from tensorflow.keras import activations
from tensorflow.keras import backend
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

# from tensorflow.keras.engine.base_layer import Layer


class TALSTMCell(tf.keras.layers.Layer):
    """
    Time Aware LSTM Cell
    """

    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        **kwargs,
    ):
        if units < 0:
            raise ValueError(
                f"Received an invalid value for units, expected "
                f"a positive integer, got {units}."
            )
        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        else:
            self._enable_caching_device = kwargs.pop("enable_caching_device", False)
        super(TALSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        implementation = kwargs.pop("implementation", 1)
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation
        self.state_size = [self.units, self.units]
        self.output_size = self.units

    def build(self, input_shape):
        """
        Required to add weights that are input shape dependent
        """
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1] - 2
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device,
        )
        # time related
        # W_xdelta, W_tdelta, W_xs and W_ts, Eqs. (9) & (10)
        self.time_kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name="time_kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )

        # W_delta and W_s for Eqs.(7) & (8)
        self.time_kernel2 = self.add_weight(
            shape=(2, input_dim),
            name="time_kernel2",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )

        # W_delta,o and W_so for Eq.(12)
        self.time_kernel3 = self.add_weight(
            shape=(input_dim, self.units * 2),
            name="time_kernel3",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )
        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return backend.concatenate(
                        [
                            self.bias_initializer((self.units,), *args, **kwargs),
                            initializers.get("ones")((self.units,), *args, **kwargs),
                            self.bias_initializer((self.units * 2,), *args, **kwargs),
                        ]
                    )

            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
            # time related
            self.time_bias1 = self.add_weight(
                shape=(input_dim * 2,),
                name="time_bias1",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
            self.time_bias2 = self.add_weight(
                shape=(self.units * 2,),
                name="time_bias2",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        #         dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        #         rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        #             h_tm1, training, count=4)

        # take out the time features
        tf1 = tf.expand_dims(inputs[:, -1], -1)
        tf2 = tf.expand_dims(inputs[:, -2], -1)
        inputs = inputs[:, :-2]

        if self.implementation == 1:
            if 0 < self.dropout < 1.0:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
                inputs_d = inputs  # delta
                inputs_s = inputs  # s
            k_i, k_f, k_c, k_o = tf.split(self.kernel, num_or_size_splits=4, axis=1)
            W_xd, W_xs, W_td, W_ts = tf.split(
                self.time_kernel, num_or_size_splits=4, axis=1
            )
            W_d, W_s = tf.split(self.time_kernel2, num_or_size_splits=2, axis=0)
            x_i = backend.dot(inputs_i, k_i)
            x_f = backend.dot(inputs_f, k_f)
            x_c = backend.dot(inputs_c, k_c)
            x_o = backend.dot(inputs_o, k_o)
            # time related
            delta_tk = backend.dot(tf1, W_d)
            s_tk = backend.dot(tf2, W_s)
            T_d = backend.dot(inputs_d, W_xd)  # T_delta
            T_s = backend.dot(inputs_s, W_xs)  # T_s
            if self.use_bias:
                b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)

                x_i = backend.bias_add(x_i, b_i)
                x_f = backend.bias_add(x_f, b_f)
                x_c = backend.bias_add(x_c, b_c)
                x_o = backend.bias_add(x_o, b_o)

                b_d, b_s = tf.split(self.time_bias1, num_or_size_splits=2, axis=0)
                delta_tk = backend.bias_add(delta_tk, b_d)
                s_tk = backend.bias_add(s_tk, b_s)

                b_td, b_ts = tf.split(self.time_bias2, num_or_size_splits=2, axis=0)
                T_d = backend.bias_add(T_d, b_td)
                T_s = backend.bias_add(T_s, b_ts)

            delta_tk = self.activation(delta_tk)  # complete
            s_tk = self.activation(s_tk)  # complete

            T_d = T_d + backend.dot(delta_tk, W_td)
            T_s = T_s + backend.dot(s_tk, W_ts)
            T_delta = self.recurrent_activation(T_d)
            T_s = self.recurrent_activation(T_s)

            # Eq.(12)
            W_do, W_so = tf.split(self.time_kernel3, num_or_size_splits=2, axis=1)
            x_o = x_o + backend.dot(delta_tk, W_do)
            x_o = x_o + backend.dot(s_tk, W_so)

            if 0 < self.recurrent_dropout < 1.0:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1, T_delta, T_s)
        else:
            if 0.0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]
            z = backend.dot(inputs, self.kernel)
            z += backend.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = backend.bias_add(z, self.bias)

            z = tf.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]

    def _compute_carry_and_output(self, x, h_tm1, c_tm1, T_delta, T_s):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + backend.dot(h_tm1_i, self.recurrent_kernel[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f
            + backend.dot(
                h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2]
            )
        )
        c = f * T_delta * c_tm1 + i * T_s * self.activation(
            x_c
            + backend.dot(
                h_tm1_c, self.recurrent_kernel[:, self.units * 2 : self.units * 3]
            )
        )
        o = self.recurrent_activation(
            x_o + backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
        )
        return c, o


def _caching_device(rnn_cell):
    """Returns the caching device for the RNN variable.
    This is useful for distributed training, when variable is not located as same
    device as the training worker. By enabling the device cache, this allows
    worker to read the variable once and cache locally, rather than read it every
    time step from remote when it is needed.
    Note that this is assuming the variable that cell needs for each time step is
    having the same value in the forward path, and only gets updated in the
    backprop. It is true for all the default cells (SimpleRNN, GRU, LSTM). If the
    cell body relies on any variable that gets updated every time step, then
    caching device will cause it to read the stall value.
    Args:
    rnn_cell: the rnn cell instance.
    """
    if tf.executing_eagerly():
        # caching_device is not supported in eager mode.
        return None
    if not getattr(rnn_cell, "_enable_caching_device", False):
        return None
    # Don't set a caching device when running in a loop, since it is possible that
    # train steps could be wrapped in a tf.while_loop. In that scenario caching
    # prevents forward computations in loop iterations from re-reading the
    # updated weights.
    if control_flow_util.IsInWhileLoop(tf.compat.v1.get_default_graph()):
        logging.warning(
            "Variable read device caching has been disabled because the "
            "RNN is in tf.while_loop loop context, which will cause "
            "reading stalled value in forward path. This could slow down "
            "the training due to duplicated variable reads. Please "
            "consider updating your code to remove tf.while_loop if possible."
        )
        return None
    if rnn_cell._dtype_policy.compute_dtype != rnn_cell._dtype_policy.variable_dtype:
        logging.warning(
            "Variable read device caching has been disabled since it "
            "doesn't work with the mixed precision API. This is "
            "likely to cause a slowdown for RNN training due to "
            "duplicated read of variable for each timestep, which "
            "will be significant in a multi remote worker setting. "
            "Please consider disabling mixed precision API if "
            "the performance has been affected."
        )
        return None
    # Cache the value on the device that access the variable.
    return lambda op: op.device
