from abc import abstractmethod

import numpy as np



class Optimizer(
    # Use to serialize subclasses functions, now not need use

    # Example
    # metaclass=SerializeMetaClass,
    # id_field='name',
    # saved_fields = ['ser_fn'],
):
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

    def clear_state(self):
        pass

    def zero_grad(self):
        self.parameters = []
        self.gradients = []

    @abstractmethod
    def next_step(self):
        pass

