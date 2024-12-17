from v1.src.base.loss_function import LossFunction
from v1.src.base.metrics.metric import Metric


class LossMetric(Metric):
    def __init__(
            self,
            loss_function: LossFunction = None,
            published_name: str = "average_loss",
    ):
        super().__init__(
            name=type(self).__name__
        )
        self.__loss_function = loss_function

        self.iterations = 0
        self.overall_loss = 0.

        self.last_iterations = 0
        self.last_overall_loss = 0.

        self.published_name = published_name

    @property
    def average_loss(self):
        if self.iterations == 0:
            return float('nan')
        return self.overall_loss / self.iterations

    def clear_state(self):
        self.last_iterations = self.iterations
        self.last_overall_loss = self.overall_loss

        self.iterations = 0
        self.overall_loss = 0.

    def update_state(self, y_pred, e):
        self.iterations += 1
        self.overall_loss += self.__loss_function(y_pred=y_pred, e=e)

    def get_metric_state(self):
        result = {
            'name': LossMetric.__name__,
            'loss_function': self.__loss_function.name,
            'iterations': self.iterations,
            'overall_loss': self.overall_loss,
            'average_loss': self.average_loss,
            self.published_name: self.average_loss,
        }
        return result

    def get_published_value(self):
        return f'{self.published_name}: {self.average_loss}'