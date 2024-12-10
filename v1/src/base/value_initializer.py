from typing import Callable
import numpy as np

from v1.src.serialize import SerializeMetaClass


class ValueInitializer(
    metaclass=SerializeMetaClass,
    id_field='name',
    saved_fields = ['__init_function'],
):
    def __init__(
            self,
            init_function: Callable[[np.array], np.array],
            name: str
    ):
        self.__init_function = init_function
        self.name = name

    def __call__(self, shape):
        create_array = self.__init_function(shape)

        if create_array is not None and create_array.ndim == 1:
            create_array = np.expand_dims(create_array, axis=0)

        return create_array

def zero_initializer() -> ValueInitializer:
    def zero_init_function(shape) -> np.array:
        return np.zeros(shape)

    return ValueInitializer(zero_init_function, name="zero_initializer")

def uniform_initializer(min_value: float = -0.5, max_value: float = 0.5) -> ValueInitializer:
    def uniform_init_function(shape) -> np.array:
        return np.random.uniform(min_value, max_value, shape)

    return ValueInitializer(uniform_init_function, name="uniform_initializer")

def none_weights_initializer() -> ValueInitializer:
    def none_weights_init_function(_) -> np.array:
        return None

    return ValueInitializer(none_weights_init_function, name="none_weights_initializer")

def he_initializer() -> ValueInitializer:
    def he_init_function(shape) -> np.array:
        n_in, n_out = shape
        return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

    return ValueInitializer(he_init_function, name="he_initializer")

def xavier_initializer() -> ValueInitializer:
    def xavier_init_function(shape) -> np.array:
        n_in, n_out = shape
        return np.random.randn(n_in, n_out) * np.sqrt(1 / (n_in + n_out))

    return ValueInitializer(xavier_init_function, name="xavier_initializer")

def orthogonal_initializer() -> ValueInitializer:
    def orthogonal_init_function(shape) -> np.array:
        n_in, n_out = shape
        return np.linalg.qr(np.random.randn(n_in, n_out))[0]

    return ValueInitializer(orthogonal_init_function, name="orthogonal_initializer")

def torch_rnn_initializer() -> ValueInitializer:
    def torch_rnn_init_function(shape) -> np.array:
        if isinstance(shape, int):
            hidden_size = shape
        else:
            hidden_size = shape[-1]
        return np.random.random(shape) * np.sqrt(1 / hidden_size)

    return ValueInitializer(torch_rnn_init_function, name="torch_rnn_initializer")

ValueInitializer.__register_generators__([
    zero_initializer, uniform_initializer, none_weights_initializer, he_initializer,
    xavier_initializer, orthogonal_initializer, torch_rnn_initializer
])