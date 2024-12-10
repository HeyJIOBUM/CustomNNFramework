from v1.src.base.activation import *
from v1.src.base.callbacks.default_callbacks import ProgressBarCallback, ModelSaveCallback
from v1.src.base.callbacks.default_callbacks.early_stopping_callback import EarlyStoppingCallback
from v1.src.base.data import data_augmentation, ModelDataSource
from v1.src.base.layers import InputLayer, LinearLayer
from v1.src.base.loss_function import mse
from v1.src.base.metrics import AccuracyMetric, LossMetric
from v1.src.base.metrics.matching_functon import one_hot_matching_function
from v1.src.base.models import SequentialModel
from v1.src.base.optimizers.adam import Adam
from v1.src.base.value_initializer import he_initializer, xavier_initializer
from v1.src.mnist.mnist_dataloader import MnistDataloader

mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


batch_size = 200

data_source = ModelDataSource(
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    train_data_augmentations=[
        data_augmentation.scaling_pil(
            copies=1,
            scale_range=(0.5, 2)
        ),
        data_augmentation.cropping(
            copies=1,
            cropped_part_range=(0.5, 0.9)
        ),
    ],
    data_augmentations=[
        data_augmentation.flatten(),
        data_augmentation.normalize(),
        data_augmentation.one_hot_labels(num_classes=10)
    ],
    batch_size=batch_size,
)

model = SequentialModel(
    layers=[
        InputLayer(784),
        LinearLayer(256,
                    activation=relu(),
                    prev_weights_initializer=xavier_initializer(),
                    ),
        LinearLayer(64,
                    activation=relu(),
                    prev_weights_initializer=xavier_initializer(),
                    ),
        LinearLayer(10,
                    activation=linear(),
                    prev_weights_initializer=xavier_initializer(),
                    ),
    ]
)

model.build(
    loss_function=mse(),
    optimizer=Adam(learning_rate=0.0013),
    metrics=[
        AccuracyMetric(
            matching_function=one_hot_matching_function()
        ),
        LossMetric(
            loss_function=mse(),
            published_name='mse_loss'
        ),
    ],
)

model.fit(
    model_data_source=data_source,
    epochs=25,
    callbacks=[
        ProgressBarCallback(
            count_mode='batch',
            monitors=[
                'accuracy',
                'mse_loss',
            ]),
        EarlyStoppingCallback(
            monitor='accuracy',
            mode='max',
        ),
        ModelSaveCallback(
            monitor='accuracy',
            mode='max',
            monitor_save_threshold=0.98,
            filepath='mnist_greater_0.98accuracy',
        )
    ],
)

