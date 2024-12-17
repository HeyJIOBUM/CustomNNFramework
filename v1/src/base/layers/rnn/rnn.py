import numpy as np

from v1.src.base.activation import Activation
from v1.src.base.layers.rnn.rnn_base import RNNBase
from v1.src.base.optimizers import Optimizer
from v1.src.base.value_initializer import ValueInitializer, torch_rnn_initializer


class RNNLayer(RNNBase):
    def __init__(
            self,
            neurons: int,
            activation: Activation = None,
            is_trainable: bool = True,
            prev_weights_initializer: ValueInitializer = torch_rnn_initializer(),
            recurrent_weights_initializer: ValueInitializer = torch_rnn_initializer(),

            stacked_layers: int = 1,

            use_bias: bool = True,
            bias_initializer: ValueInitializer = torch_rnn_initializer(),
    ):
        super().__init__(
            neurons=neurons,
            activation=activation,
            is_trainable=is_trainable,
            prev_weights_initializer=prev_weights_initializer,
            recurrent_weights_initializer=recurrent_weights_initializer,
            stacked_layers=stacked_layers,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
        )

        self.prev_in = None
        self.prev_s = None
        self.prev_out = None
        self.prev_h_t_minus_1 = None

    @property
    def h_t(self):
        return self.state

    @h_t.setter
    def h_t(self, h_t):
        self.state = h_t

    def forward(self, in_batch: np.array, initial_state = None, training = True) -> np.array:
        # transpose in_batch from (batch_size, timesteps, features) to (timesteps, batch_size, features)
        in_batch = np.transpose(in_batch, (1, 0, 2))

        timesteps = in_batch.shape[0]
        batch_size = in_batch.shape[1]

        if initial_state is not None:
            self.h_t = initial_state
        else:
            self.h_t = self.get_initial_state((batch_size, self.neurons))

        output = []
        self.prev_in = [[[] for _ in range(timesteps)] for _ in range(self.stacked_layers)]
        self.prev_s = [[[] for _ in range(timesteps)] for _ in range(self.stacked_layers)]
        self.prev_out = [[[] for _ in range(timesteps)] for _ in range(self.stacked_layers)]
        self.prev_h_t_minus_1 = [[[] for _ in range(timesteps)] for _ in range(self.stacked_layers)]

        for timestep in range(timesteps):
            in_batch_sequence = in_batch[timestep]
            for layer in range(self.stacked_layers):
                self.prev_in[layer][timestep] = in_batch_sequence

                h_t_minus_1 = self.h_t[layer]
                self.prev_h_t_minus_1[layer][timestep] = h_t_minus_1

                self.h_t[layer] = np.matmul(in_batch_sequence, self.prev_weights[layer])
                self.h_t[layer] += np.matmul(h_t_minus_1, self.recurrent_weights[layer])

                if self.bias[layer] is not None:
                    self.h_t[layer] -= self.bias[layer]

                self.prev_s[layer][timestep] = self.h_t[layer]

                if self.activation is not None:
                    self.h_t[layer] = np.array([
                        self.activation(unbatched_value) for unbatched_value in self.h_t[layer]
                    ])

                self.prev_out[layer][timestep] = self.h_t[layer]

                in_batch_sequence = self.h_t[layer]
            output.append(self.h_t[-1])

        output = np.array(output).transpose((1,0,2))

        return output

    def backward(self, layer_gradient: np.array, optimizer: Optimizer) -> np.array:
        # transpose layer_gradient from (batch_size, timesteps, out_features) to (timesteps, batch_size, out_features)
        layer_gradient = np.transpose(layer_gradient, (1, 0, 2))

        timesteps = layer_gradient.shape[0]
        batch_size = layer_gradient.shape[1]

        de_dy = layer_gradient
        de_ds = [[] for _ in range(timesteps)]

        for layer in reversed(range(self.stacked_layers)):
            for timestep in reversed(range(timesteps)):
                prev_layer_s = self.prev_s[layer][timestep]
                dh_t_ds_t = np.apply_along_axis(
                    self.activation.jacobian,
                    axis=prev_layer_s.ndim - 1,
                    arr=prev_layer_s,
                )

                de_dy_t = de_dy[timestep]

                if timestep != timesteps - 1:
                    ds_t_plus_1_dh_t =  np.transpose(self.recurrent_weights[layer])

                    de_dy_t += np.dot(de_ds[timestep + 1], ds_t_plus_1_dh_t)

                de_ds[timestep] = np.einsum('bf,bfj->bf',
                                  de_dy_t,
                                  dh_t_ds_t,
                                  optimize='optimal',
                                  )

            de_ds = np.array(de_ds)

            self.__adjust_stacked_layer_params(optimizer, layer, de_ds)

            ds_dx = np.transpose(self.prev_weights[layer])
            de_dx = np.dot(de_ds, ds_dx)

            # for prev layer de_dx equals de_dy
            de_dy = de_dx

        return np.transpose(de_dy, (1, 0, 2))

    def __adjust_stacked_layer_params(self, optimizer: Optimizer, stacked_layer, de_ds):
        self.__adjust_bias(optimizer, stacked_layer, de_ds)
        self.__adjust_prev_weights(optimizer, stacked_layer, de_ds)
        self.__adjust_recurrent_weights(optimizer, stacked_layer, de_ds)

    def __adjust_bias(self, optimizer: Optimizer, stacked_layer, de_ds):
        if not self.use_bias or not self.is_trainable:
            return

        bias_gradient = - de_ds.sum(axis=tuple(range(de_ds.ndim - 1)))
        optimizer(param=self.bias[stacked_layer], param_gradient=bias_gradient)

    def __adjust_prev_weights(self, optimizer: Optimizer, stacked_layer, de_ds):
        if self.prev_weights is None or not self.is_trainable:
            return

        prev_layer_h = np.array(self.prev_in[stacked_layer])


        weights_gradient = np.einsum('tlj,tlk->kj',
                                     de_ds,
                                     prev_layer_h,
                                     optimize='optimal',
                                     )

        optimizer(param=self.prev_weights[stacked_layer], param_gradient=weights_gradient)

    def __adjust_recurrent_weights(self, optimizer: Optimizer, stacked_layer, de_ds):
        if self.recurrent_weights is None or not self.is_trainable:
            return

        prev_layer_in = np.array(self.prev_h_t_minus_1[stacked_layer])

        weights_gradient = np.einsum('tlj,tlk->kj',
                                     de_ds,
                                     prev_layer_in,
                                     optimize='optimal',
                                     )

        optimizer(param=self.recurrent_weights[stacked_layer], param_gradient=weights_gradient)
