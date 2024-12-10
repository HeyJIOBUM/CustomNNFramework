from v1.src.base.callbacks.default_callbacks import ProgressBarCallback
from v1.src.base.data import data_augmentation, ModelDataSource
from v1.src.base.models.model import Model
from v1.src.mnist.mnist_dataloader import MnistDataloader

mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

batch_size = 200

data_source = ModelDataSource(
    test_data=(x_test, y_test),
    data_augmentations=[
        data_augmentation.flatten(),
        data_augmentation.normalize(),
        data_augmentation.one_hot_labels(num_classes=10)
    ],
    batch_size=batch_size,
)

model = Model.load_from_file("mnist_greater_0.98accuracy")

model.test_epoch(
    test_data=data_source.test_data_batches(1),
    callbacks=[
        ProgressBarCallback(
            count_mode='batch',
            monitors=[
                'accuracy',
                'mse_loss',
            ]),
    ]
)
