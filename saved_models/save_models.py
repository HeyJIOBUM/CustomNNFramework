from v1.src.base.activation import *
from v1.src.base.callbacks.default_callbacks import ProgressBarCallback
from v1.src.base.data import data_augmentation, ModelDataSource
from v1.src.base.layers import InputLayer
from v1.src.base.layers.linear_layer import *
from v1.src.base.loss_function import mse
from v1.src.base.metrics import AccuracyMetric, one_hot_matching_function
from v1.src.base.models import SequentialModel
from v1.src.base.optimizers.adam import Adam
from v1.src.base.value_initializer import he_initializer
from v1.src.mnist.mnist_dataloader import MnistDataloader

mnist_dataloader = MnistDataloader(
    training_images_filepath = "../v1/src/mnist/data/train-images-idx3-ubyte/train-images-idx3-ubyte",
    training_labels_filepath = "../v1/src/mnist/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
    test_images_filepath = "../v1/src/mnist/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
    test_labels_filepath = "../v1/src/mnist/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

batch_size = 200

data_source = ModelDataSource(
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    data_augmentations=[
        data_augmentation.flatten(),
        data_augmentation.normalize(),
        data_augmentation.one_hot_labels(num_classes=10)
    ],
    shuffle=True,
    batch_size=batch_size,
)

model = SequentialModel(
    layers=[
        InputLayer(784),
        LinearLayer(256,
                    activation=relu(),
                    prev_weights_initializer=he_initializer(),
                    ),
        LinearLayer(50,
                    activation=relu(),
                    prev_weights_initializer=he_initializer(),
                    ),
        LinearLayer(10,
                    activation=linear(),
                    prev_weights_initializer=he_initializer(),
                    ),
    ]
)

print(f"learning_rate: {0.0013}")

model.build(
    loss_function=mse(),
    optimizer=Adam(learning_rate=0.0013),
    metrics=[
        AccuracyMetric(matching_function=one_hot_matching_function())
    ]
)

model.fit(
    model_data_source=data_source,
    epochs=10,
    callbacks=[
        ProgressBarCallback(),
    ],
)

model.save_to_file("2_hl_Adam_lr0.0013.txt")