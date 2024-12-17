from abc import abstractmethod

import numpy as np

from v1.src.base.activation import Activation
from v1.src.base.layers import Layer
from v1.src.base.optimizers import Optimizer
from v1.src.base.value_initializer import ValueInitializer, zero_initializer, orthogonal_initializer, he_initializer


class RNNBase(Layer):
    def __init__(
            self,
            neurons: int,
            activation: Activation = None,
            is_trainable: bool = True,
            prev_weights_initializer: ValueInitializer = he_initializer(),
            recurrent_weights_initializer: ValueInitializer = orthogonal_initializer(),

            stacked_layers: int = 1,

            use_bias: bool = True,
            bias_initializer: ValueInitializer = zero_initializer(),
    ):
        super().__init__(
            neurons=neurons,
            activation=activation,
            is_trainable=is_trainable,
            prev_weights_initializer=prev_weights_initializer,
            name=RNNBase.__name__,
        )
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias = None

        self.stacked_layers = stacked_layers

        self.recurrent_weights_initializer = recurrent_weights_initializer
        self.recurrent_weights = None

        self.state = None

    @abstractmethod
    def forward(self, in_batch: np.array, training=True) -> np.array:
        pass

    @abstractmethod
    def backward(self, layer_gradient: np.array, optimizer: Optimizer) -> np.array:
        pass

    def get_initial_state(self, shape: ()):
        state = []
        for _ in range(self.stacked_layers):
            state.append(np.zeros(shape))
        return state

    def reset_state(self):
        if self.state:
            for layer in range(self.stacked_layers):
                for state_param in range(self.state[layer]):
                    self.state[layer][state_param] = np.zeros_like(self.state[layer][state_param])

    def init_layer_params(self, prev_layer_neurons, reassign_existing=True):
        if self.prev_weights is None or reassign_existing:
            self.prev_weights = []
            self.prev_weights.append(self.prev_weights_initializer((prev_layer_neurons, self.neurons)))
            for layer in range(1, self.stacked_layers):
                self.prev_weights.append(self.prev_weights_initializer((self.neurons, self.neurons)))

        if self.recurrent_weights is None or reassign_existing:
            self.recurrent_weights = []
            for layer in range(self.stacked_layers):
                self.recurrent_weights.append(self.recurrent_weights_initializer((self.neurons, self.neurons)))

        if self.use_bias and (reassign_existing or self.bias is None):
            self.bias = []
            for layer in range(self.stacked_layers):
                self.bias.append(self.bias_initializer(self.neurons))

        if reassign_existing:
            self.state = None

    def __getstate__(self):
        state = super().__getstate__()
        state['use_bias'] = self.use_bias
        state['bias'] = self.bias
        state['bias_initializer'] = self.bias_initializer
        state['recurrent_weights'] = self.recurrent_weights
        state['recurrent_weights_initializer'] = self.recurrent_weights_initializer
        state['stacked_layers'] = self.stacked_layers
        state['state'] = self.state
        return state


    def __setstate__(self, state):
        super().__setstate__(state)
        self.use_bias = state['use_bias']
        self.bias = state['bias']
        self.bias_initializer = state['bias_initializer']
        self.recurrent_weights = state['recurrent_weights']
        self.recurrent_weights_initializer = state['recurrent_weights_initializer']
        self.stacked_layers = state['stacked_layers']
        self.state = state['state']
