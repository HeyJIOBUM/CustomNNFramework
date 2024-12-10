import matplotlib.pyplot as plt

from v1.src.base.data import ModelDataSource, data_augmentation
from v1.src.mnist.mnist_dataloader import MnistDataloader

mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


data_source = ModelDataSource(
    test_data=(x_test[:5], y_test[:5]),
    data_augmentations=[
        data_augmentation.scaling_pil(
            copies=1,
            scale_range=(0.5,2)
        ),
        data_augmentation.cropping(
            copies=1,
            cropped_part_range=(0.5,0.9)
        ),
    ],
    shuffle=False,
    batch_size=20,
)

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(15, 15))
    index = 1
    for x, _ in zip(images, title_texts):
        image = x
        plt.subplot(rows, cols, index)
        plt.imshow(image)
        index += 1

    plt.show()

first_batch = data_source.test_data_batches()[0]
x_batch, y_bach = first_batch

show_images(x_batch, y_bach)
