from time import sleep

import numpy as np
from tqdm import tqdm

from v1.src.base.callbacks import Callback
from v1.src.base.callbacks.callback_list import CallbackList
from v1.src.base.data.data_batch_wrapper import DataBatchWrapper
from v1.src.base.data.model_data_source import ModelDataSource
from v1.src.base.layers.layer import Layer
from v1.src.base.loss_function import LossFunction, mse
from v1.src.base.metrics import LossMetric, MetricList
from v1.src.base.metrics.metric import Metric
from v1.src.base.models.sequential_model import SequentialModel
from v1.src.base.optimizers.optimizer import Optimizer
from v1.src.base.optimizers.sgd import SGD


class CustomSequentialModel(SequentialModel):
    def __init__(self,
                 layers: [Layer] = None,
                 optimizer: Optimizer = SGD(),
                 loss_function: LossFunction = mse(),
                 metrics: [Metric] or MetricList = None,
                 ):
        super().__init__(
            layers=layers,
            optimizer=optimizer,
            loss_function=loss_function,
            metrics=metrics,
            name=type(self).__name__
        )

    def fit(self,
            model_data_source: ModelDataSource,
            epochs: int = 1,
            callbacks: CallbackList or [Callback] = None,
            disable_tqdm = False):
        train_data = model_data_source.train_data_batches()
        test_data = model_data_source.test_data_batches()
        for epoch in range(epochs):
            print(f'epoch: {epoch+1}')
            sleep(0.05)
            self.train_epoch(train_data, disable_tqdm=disable_tqdm)
            self.test_epoch(test_data, disable_tqdm=disable_tqdm)

    def train_epoch(
            self,
            train_data: DataBatchWrapper,
            callbacks: CallbackList or [Callback] = None,
            disable_tqdm = False):
        if len(train_data) == 0:
            return

        self.metrics.clear_state()
        for i, (x_batch, e_batch) in enumerate(pbar := tqdm(train_data,
                                                    total=len(train_data),
                                                    desc="train",
                                                    disable=disable_tqdm)):
            self.optimizer.zero_grad()

            y_pred_batch = self.forward(x_batch)

            loss_gradient_batch = np.array([
                self.loss_function.gradient(y_pred=y_pred, e=e)
                for y_pred, e in zip(y_pred_batch, e_batch)
            ])

            self.backward(loss_gradient_batch=loss_gradient_batch)

            [self.metrics.update_state(y, e) for y, e in zip(y_pred_batch, e_batch)]

            pbar.set_postfix_str(self.metrics.get_metric_value()[0], refresh=True)

            self.optimizer.next_step()

    def test_epoch(
            self,
            test_data: DataBatchWrapper,
            callbacks: CallbackList or [Callback] = None,
            disable_tqdm = False):
        if len(test_data) == 0:
            return

        self.metrics.clear_state()
        for i, (x_batch, e_batch) in enumerate(pbar := tqdm(test_data,
                                                    total=len(test_data),
                                                    desc="test",
                                                    disable=disable_tqdm)):
            y_batch = self.forward(x_batch)
            [self.metrics.update_state(y, e) for y, e in zip(y_batch, e_batch)]

            pbar.set_postfix_str(self.metrics.get_metric_value()[0], refresh=True)