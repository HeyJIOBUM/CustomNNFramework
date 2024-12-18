from abc import abstractmethod

import numpy as np


class Optimizer:
    def __init__(
            self,
            name: str,
            learning_rate: float,
            ):

        self.name = name
        self.learning_rate = learning_rate

        self.parameters = []
        self.gradients = []

    def __call__(self, param: np.array, param_gradient: np.array):
        self.parameters.append(param)
        self.gradients.append(param_gradient)

    def zero_grad(self):
        self.parameters = []
        self.gradients = []

    @abstractmethod
    def next_step(self):
        pass

