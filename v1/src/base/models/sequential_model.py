from abc import ABC

import numpy as np

from v1.src.base.callbacks import Callback
from v1.src.base.callbacks.callback_list import CallbackList
from v1.src.base.data import ModelDataSource, DataBatchWrapper
from v1.src.base.layers.layer import Layer
from v1.src.base.loss_function import LossFunction
from v1.src.base.metrics.metric import Metric
from v1.src.base.metrics.metric_list import MetricList
from v1.src.base.models.model import Model
from v1.src.base.optimizers.optimizer import Optimizer


class SequentialModel(Model, ABC):
    def __init__(self,
                 layers: [Layer] = None,
                 optimizer: Optimizer = None,
                 loss_function: LossFunction = None,
                 metrics: [Metric] or MetricList = None,
                 name: str = 'SequentialModel',
                 ):
        super().__init__(
            layers=layers,
            optimizer=optimizer,
            loss_function=loss_function,
            metrics=metrics,
            name=name,
        )

    def fit(self,
            model_data_source: ModelDataSource,
            epochs: int = 1,
            callbacks: CallbackList or [Callback] = None,
            ):

        train_data = model_data_source.train_data_batches()
        test_data = model_data_source.test_data_batches()

        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                model=self,
                callbacks=callbacks,
                epochs=epochs,
                test_batches=len(test_data),
                train_batches=len(train_data),
                batch_size=model_data_source.batch_size,
            )

        callbacks.on_fit_start(epochs)
        for epoch in range(epochs):
            callbacks.on_epoch_start(epoch)

            self.train_epoch(train_data, callbacks=callbacks, epoch=epoch)
            self.test_epoch(test_data, callbacks=callbacks, epoch=epoch)

            callbacks.on_epoch_end(epoch)

            # stop if stop_training flag was set in callbacks
            if self.stop_training:
                break
        callbacks.on_fit_end()

    def train_epoch(
            self,
            train_data: DataBatchWrapper,
            callbacks: CallbackList or [Callback] = None,
            epoch: int = 1,
    ):
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                model=self,
                callbacks=callbacks,
                train_batches=len(train_data),
                batch_size=train_data.batch_size,
            )

        callbacks.on_train_epoch_start(epoch)

        self.optimizer.clear_state()
        self.metrics.clear_state()
        for i, (x_batch, e_batch) in enumerate(train_data):
            self.optimizer.zero_grad()

            callbacks.on_train_batch_start(i)

            y_pred_batch = self.forward(x_batch, training=True)

            loss_gradient_batch = np.array([
                self.loss_function.gradient(y_pred=y_pred, e=e)
                for y_pred, e in zip(y_pred_batch, e_batch)
            ])

            self.backward(loss_gradient_batch=loss_gradient_batch)

            [self.metrics.update_state(y, e) for y, e in zip(y_pred_batch, e_batch)]

            self.optimizer.next_step()

            callbacks.on_train_batch_end(i, self.metrics.get_metric_state())

            # stop if stop_training flag was set in on_train_batch_end callbacks
            if self.stop_training:
                break

        callbacks.on_train_epoch_end(epoch, self.metrics.get_metric_state())

    def test_epoch(
            self,
            test_data: DataBatchWrapper,
            callbacks: CallbackList or [Callback] = None,
            epoch: int = 1,
    ):
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(
                model=self,
                callbacks=callbacks,
                test_batches=len(test_data),
                batch_size=test_data.batch_size,
            )

        callbacks.on_test_epoch_start(epoch)

        self.metrics.clear_state()
        for i, (x_batch, e_batch) in enumerate(test_data):
            callbacks.on_test_batch_start(i)

            y_batch = self.forward(x_batch, training=False)
            [self.metrics.update_state(y, e) for y, e in zip(y_batch, e_batch)]

            callbacks.on_test_batch_end(i, self.metrics.get_metric_state())

        callbacks.on_test_epoch_end(epoch, self.metrics.get_metric_state())

    def forward(self, in_batch: np.ndarray, training=True) -> np.ndarray:
        if in_batch.ndim == 1:
            in_batch = np.expand_dims(in_batch, axis=0)

        for layer in self.layers[1:]:
            in_batch = layer.forward(in_batch)
        return in_batch

    def backward(self, loss_gradient_batch: np.array):
        if loss_gradient_batch.ndim == 1:
            loss_gradient_batch = np.expand_dims(loss_gradient_batch, axis=0)

        for layer in reversed(self.layers[1:]):
            loss_gradient_batch = layer.backward(loss_gradient_batch, optimizer=self.optimizer)

    def add_layer(self, layer: Layer):
        layer.init_layer_params(
            prev_layer_neurons=self.layers[-1].neurons,
            reassign_existing=True
        )
        self.layers.append(layer)

    def init_layers_params(self, reassign_existing=True):
        for i in range(1, len(self.layers)):
            self.layers[i].init_layer_params(
                prev_layer_neurons=self.layers[i-1].neurons,
                reassign_existing=reassign_existing
            )
