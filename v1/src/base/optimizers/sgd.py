import numpy as np

from v1.src.base.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
            self,
            learning_rate: float = 0.001,
            momentum: float = 0.0,
            nesterov: bool = False,
    ):
        super().__init__(
            name='SGD',
            learning_rate=learning_rate
        )
        self.momentum_coefficient = momentum
        self.momentums = None

        self.nesterov = nesterov

    def next_step(self):
        if self.momentum_coefficient == 0.0:
            for param, gradient in zip(self.parameters, self.gradients):
                param -= np.multiply(self.learning_rate, gradient)
        else:
            if self.momentums is None:
                self.momentums = [np.zeros(gradient.shape) for gradient in self.gradients]

            if self.nesterov:
                for param, gradient, saved_momentum in zip(self.parameters, self.gradients, self.momentums):
                    saved_momentum *= self.momentum_coefficient
                    saved_momentum -= np.multiply(self.learning_rate, gradient)

                    param += np.multiply(self.momentum_coefficient, saved_momentum)
                    param -= np.multiply(self.learning_rate, gradient)
            else:
                for param, gradient, saved_momentum in zip(self.parameters, self.gradients, self.momentums):
                    saved_momentum *= self.momentum_coefficient
                    saved_momentum -= np.multiply(self.learning_rate, gradient)

                    param += saved_momentum
