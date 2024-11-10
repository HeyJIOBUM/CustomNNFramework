from v1.src.base.layers.layer import Layer
from v1.src.base.optimizers.optimizer import Optimizer
from v1.src.base.value_initializer import *

class DropoutLayer(Layer):
    def __init__(
            self,
            neurons: int,
            rate: float,
    ):
        super().__init__(
            neurons=neurons,
            prev_weights_initializer=none_weights_initializer(),
            activation=None,
            is_trainable=False,
            name=self.__class__.__name__,
        )
        self.rate = rate
        self.masks = None

    def backward(self, layer_gradient: np.array, optimizer: Optimizer):
        return layer_gradient * self.masks

    def forward(self, in_batch: np.array, training=True) -> np.array:
        if training:
            self.masks = (np.random.binomial(n=1, p=(1 - self.rate), size=in_batch.shape[1] * in_batch.shape[0])
                          .reshape(in_batch.shape[0], -1))
            in_batch = np.multiply(in_batch, self.masks)

        return in_batch


    def __getstate__(self):
        state = super().__getstate__()
        state['rate'] = self.rate
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.rate = state['rate']
