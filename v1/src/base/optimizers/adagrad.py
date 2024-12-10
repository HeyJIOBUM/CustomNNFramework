import numpy as np

from v1.src.base.optimizers.optimizer import Optimizer


class AdaGrad(Optimizer):
    def __init__(
            self,
            learning_rate: float = 0.001,
            initial_accumulator_value=0.1,
            epsilon=1e-7,
    ):
        super().__init__(
            name='AdaGrad',
            learning_rate=learning_rate
        )
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

        self.accumulators = None

    def next_step(self):
        if self.accumulators is None:
            self.accumulators = [np.full(param.shape, self.initial_accumulator_value) for param in self.parameters]

        for param, gradient, accumulator in zip(self.parameters, self.gradients, self.accumulators):
            accumulator += gradient ** 2

            param -= (self.learning_rate * gradient / np.sqrt(accumulator + self.epsilon))
