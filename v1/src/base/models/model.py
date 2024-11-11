import pickle
from abc import ABC, abstractmethod

import numpy as np

from v1.src.base.callbacks import Callback
from v1.src.base.callbacks.callback_list import CallbackList
from v1.src.base.data.data_batch_wrapper import DataBatchWrapper
from v1.src.base.data.model_data_source import ModelDataSource
from v1.src.base.layers.layer import Layer
from v1.src.base.loss_function import LossFunction
from v1.src.base.metrics.metric import Metric
from v1.src.base.optimizers.optimizer import Optimizer


class Model(ABC):
    def __init__(self,
                 name: str,
                 layers: [Layer] = None,
                 loss_function: LossFunction = None,
                 optimizer: Optimizer = None,
                 metric: Metric = None,
                 ):
        self.name = name

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metric = metric

        self.layers = layers or []

    def build(
            self,
            loss_function: LossFunction = None,
            optimizer: Optimizer = None,
            metric: Metric = None):
        self.loss_function = loss_function or self.loss_function
        self.optimizer = optimizer or self.optimizer
        self.metric = metric or self.metric

    @abstractmethod
    def fit(self, model_data_source: ModelDataSource, epochs = 1, callbacks: CallbackList or [Callback] = None):
        pass

    @abstractmethod
    def train_epoch(self, train_data: DataBatchWrapper, callbacks: CallbackList or [Callback] = None):
        pass

    @abstractmethod
    def test_epoch(self, test_data: DataBatchWrapper, callbacks: CallbackList or [Callback] = None):
        pass

    @abstractmethod
    def forward(self, in_batch: np.array, training=True):
        pass

    @abstractmethod
    def backward(self, loss_gradient: np.array):
        pass

    @abstractmethod
    def add_layer(self, layer: Layer):
        pass

    @abstractmethod
    def init_layers_params(self, reassign_existing=True):
        pass

    @abstractmethod
    def summary(self):
        pass

    def __getstate__(self):
        state = {
            'name': self.name,
            'optimizer': self.optimizer,
            'loss_function': self.loss_function,
            'metric': self.metric,
            'layers': [
                layer for layer in self.layers
            ],
        }
        return state

    def __setstate__(self, state):
        self.name = state['name']
        self.optimizer = state['optimizer']
        self.loss_function = state['loss_function']
        self.metric = state['metric']
        self.layers = state['layers']

    def save_to_file(self, file):
        with open(file, "wb") as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load_from_file(file):
        with open(file, "rb") as fp:
            return pickle.load(fp)
