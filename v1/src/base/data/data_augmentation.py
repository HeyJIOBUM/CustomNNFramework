from PIL import Image

import random
import numpy as np
from scipy import ndimage

class DataAugmentation:
    def __init__(
            self,
            augmentate_function,
            name: str = "DataAugmentation",
    ):
        self.__augmentate_function = augmentate_function
        self.name = name
        pass

    def __call__(self, data_tuple: (np.array, np.array)) -> [(np.array, np.array)]:
        answer = self.__augmentate_function(data_tuple)
        if not isinstance(answer, list):
            return [answer]
        return answer

def __add_padding(current_img: np.array, required_shape: (int, int), padding_value: float = 0.) -> np.array:
    current_height, current_width = current_img.shape
    required_height, required_width = required_shape

    if current_height == required_height and current_width == required_width:
        return current_img

    height_to_be_filled, width_to_be_filled = (required_shape[0] - current_img.shape[0],
                                               required_shape[1] - current_img.shape[1])

    image_padding = (
        (height_to_be_filled // 2, height_to_be_filled - (height_to_be_filled // 2)),
        (width_to_be_filled // 2, width_to_be_filled - (width_to_be_filled // 2)),
    )

    return np.pad(current_img, image_padding, mode='constant', constant_values=padding_value)

def flatten():
    def augmentate_function(data_tuple):
        return np.array(data_tuple[0]).flatten(), data_tuple[1]

    return DataAugmentation(augmentate_function, name="flatten")

def normalize():
    def augmentate_function(data_tuple):
        x = np.array(data_tuple[0])
        if np.max(x) == 0:
            return [],[]
        return x / np.max(x), data_tuple[1]

    return DataAugmentation(augmentate_function, name="normalize")

def one_hot_labels(num_classes = 10):
    def augmentate_function(data_tuple):
        data, label = data_tuple
        y = np.zeros(num_classes)
        y[label-1] = 1
        return data_tuple[0], y

    return DataAugmentation(augmentate_function,name="one_hot_labels")

def scaling_pil(scale_range: (float, float) = (0.5, 1.5),
            copies: int = 2,
            padding_value: float = 0.):

    def augmentate_function(data_tuple):
        img, label = np.array(data_tuple[0]), data_tuple[1]
        img_height, img_width = img.shape

        img_normalization = np.max(img)
        pil_normalized_image = np.array(img / (img_normalization / 255), dtype=np.uint8)

        min_scale, max_scale = scale_range

        scale_values = [random.uniform(min_scale, max_scale) for _ in range(copies)]

        augmented_datas = [(img, label)]
        for scale_value in scale_values:
            new_img_height, new_img_width = int(img_height * scale_value), int(img_width * scale_value)
            new_img = (Image.fromarray(pil_normalized_image)
                         .resize((new_img_height, new_img_width)))

            if scale_value > 1:
                top_left_position = ((new_img_height - img_height) // 2, (new_img_width - img_width) // 2)
                bottom_right_position = (top_left_position[0] + img_height, top_left_position[1] + img_width)
                new_img = new_img.crop((top_left_position[1], top_left_position[0],
                              bottom_right_position[1], bottom_right_position[0]))
            new_img = np.array(new_img) / (255 / img_normalization)
            new_img = __add_padding(new_img, required_shape=img.shape, padding_value=padding_value)
            augmented_datas.append((new_img, label))
        return augmented_datas

    return DataAugmentation(augmentate_function,name="scaling_pil")

def scaling_scipy(scale_range: (float, float) = (0.5, 1.5),
            copies: int = 2,
            padding_value: float = 0.):

    def augmentate_function(data_tuple):
        img, label = np.array(data_tuple[0]), data_tuple[1]
        img_height, img_width = img.shape

        min_scale, max_scale = scale_range

        scale_values = [random.uniform(min_scale, max_scale) for _ in range(copies)]

        augmented_datas = [(img, label)]
        for scale_value in scale_values:
            new_img_height, new_img_width = int(img_height * scale_value), int(img_width * scale_value)
            new_img = ndimage.zoom(img, zoom=scale_value)

            if scale_value > 1:
                top_left_position = ((new_img_height - img_height) // 2, (new_img_width - img_width) // 2)
                bottom_right_position = (top_left_position[0] + img_height, top_left_position[1] + img_width)
                new_img = new_img[top_left_position[0]:bottom_right_position[0],
                          top_left_position[1]:bottom_right_position[1]]
            new_img = __add_padding(new_img, required_shape=img.shape, padding_value=padding_value)
            augmented_datas.append((new_img, label))
        return augmented_datas

    return DataAugmentation(augmentate_function,name="scaling_scipy")

def cropping(cropped_part_range: (float, float) = (0.7, 1),
             copies: int = 1,
             padding_value: float = 0.):
    def augmentate_function(data_tuple):
        img, label = np.array(data_tuple[0]), data_tuple[1]
        img_height, img_width = img.shape

        min_cropped_part, max_cropped_part = cropped_part_range
        cropping_values = [random.uniform(min_cropped_part, max_cropped_part) for _ in range(copies)]

        augmented_datas = [(img, label)]
        for cropping_value in cropping_values:
            cropped_height = int(cropping_value * img_height)
            cropped_width = int(cropping_value * img_width)
            top_left_position = (random.randint(0, img_height - cropped_height),
                                 random.randint(0, img_width - cropped_width))

            bottom_right_position = (top_left_position[0] + cropped_height,
                                     top_left_position[1] + cropped_width)

            cropped_img = img[top_left_position[0]:bottom_right_position[0],
                              top_left_position[1]:bottom_right_position[1]]

            cropped_img = __add_padding(current_img=cropped_img,
                                        required_shape=img.shape,
                                        padding_value=padding_value)
            augmented_datas.append((cropped_img, label))
        return augmented_datas

    return DataAugmentation(augmentate_function,name="cropping")

def rotating(rotation_range: (float, float) = (-30, 30),
             copies: int = 2,
             padding_value: float = 0.):

    def augmentate_function(data_tuple):
        img, label = np.array(data_tuple[0]), data_tuple[1]

        min_rotation, max_rotation = rotation_range
        rotation_values = [random.uniform(min_rotation, max_rotation) for _ in range(copies)]

        augmented_datas = [(img, label)]
        for rotation_value in rotation_values:
            rotated_img = ndimage.rotate(img, angle=rotation_value, reshape=False, cval=padding_value)
            augmented_datas.append((rotated_img, label))
        return augmented_datas

    return DataAugmentation(augmentate_function,name="rotating")

def horizontal_flipping():
    def augmentate_function(data_tuple):
        return [(np.flip(data_tuple[0], axis=0), data_tuple[1]), (data_tuple[0], data_tuple[1])]

    return DataAugmentation(augmentate_function,name="horizontal_flipping")

def vertical_flipping():
    def augmentate_function(data_tuple):
        return [(np.flip(data_tuple[0], axis=1), data_tuple[1]), (data_tuple[0], data_tuple[1])]

    return DataAugmentation(augmentate_function,name="vertical_flipping")

def salt_and_pepper_noise(noise_probability_range: float = (0.005, 0.01),
                          copies: int = 2):

    def augmentate_function(data_tuple):
        img, label = np.array(data_tuple[0]), data_tuple[1]
        img_height, img_width = img.shape

        img_min = np.min(img)
        img_max = np.max(img)

        min_noise, max_noise = noise_probability_range
        noise_values = [random.uniform(min_noise, max_noise) for _ in range(copies)]

        augmented_datas = [(img, label)]
        for noise_value in noise_values:
            noised_pixels_count = int(noise_value * img_height * img_width)
            noised_pixels_rows = [random.randint(0, img_height) for _ in range(noised_pixels_count)]
            noised_pixels_cols = [random.randint(0, img_width) for _ in range(noised_pixels_count)]

            noised_img = img.copy()

            for salt_pxl in zip(noised_pixels_rows[noised_pixels_count // 2:],
                                noised_pixels_cols[noised_pixels_count // 2:]):
                noised_img[salt_pxl[0], salt_pxl[1]] = img_max

            for pepper_pxl in zip(noised_pixels_rows[:noised_pixels_count // 2],
                                  noised_pixels_cols[:noised_pixels_count // 2]):
                noised_img[pepper_pxl[0], pepper_pxl[1]] = img_min

            augmented_datas.append((noised_img, label))
        return augmented_datas

    return DataAugmentation(augmentate_function,name="salt_and_pepper_noise")