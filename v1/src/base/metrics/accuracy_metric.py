from v1.src.base.metrics.matching_functon import MatchingFunction
from v1.src.base.metrics.metric import Metric


class AccuracyMetric(Metric):
    def __init__(
            self,
            matching_function: MatchingFunction = None,
            published_name: str = "accuracy",
    ):
        super().__init__(
            name=type(self).__name__
        )
        self.__matching_function = matching_function

        self.iterations = 0
        self.guessed_counter = 0

        self.last_iterations = 0
        self.last_guessed_counter = 0.

        self.published_name = published_name

    @property
    def accuracy(self):
        if self.iterations == 0:
            return float('nan')
        return self.guessed_counter / self.iterations * 100

    def clear_state(self):
        self.last_iterations = self.iterations
        self.last_guessed_counter = self.guessed_counter

        self.iterations = 0
        self.guessed_counter = 0

    def update_state(self, y_pred, e):
        self.iterations += 1
        self.guessed_counter += (self.__matching_function(y_pred, e))

    def get_metric_state(self):
        result = {
            'name': AccuracyMetric.__name__,
            'matching_function': self.__matching_function.name,
            'iterations': self.iterations,
            'guessed_counter': self.guessed_counter,
            'accuracy': self.accuracy,
            self.published_name: self.accuracy,
        }
        return result

    def get_published_value(self):
        return f'{self.published_name}: {self.accuracy}'