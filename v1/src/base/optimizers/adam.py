import numpy as np

from v1.src.base.optimizers.optimizer import Optimizer


class Adam(
    Optimizer,
    # serialized_fields=['beta_1', 'beta_2', 'epsilon']
):
    def __init__(
            self,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            # amsgrad=False,
    ):
        super().__init__(
            name='Adam',
            learning_rate=learning_rate
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # self.amsgrad = amsgrad

    def clear_state(self):
        self.parameters_m = None
        self.parameters_v = None

        # self.max_v_corrected = 0

    def next_step(self):
        if self.parameters_m is None and self.parameters_v is None:
            self.parameters_m = [np.zeros_like(param) for param in self.parameters]
            self.parameters_v = [np.zeros_like(param) for param in self.parameters]

        for param, gradient, m, v in zip(self.parameters, self.gradients, self.parameters_m, self.parameters_v):
            m *= self.beta_1
            m += (1 - self.beta_1) * gradient

            v *= self.beta_2
            v += (1 - self.beta_2) * gradient**2

            m_corrected = m / (1 - self.beta_1)
            v_corrected = v / (1 - self.beta_2)

            param -= self.learning_rate * m_corrected / np.sqrt(v_corrected + self.epsilon)

