from io import BytesIO
from typing import Union, List, Tuple
import scipy
from abc import ABC, abstractmethod
import numpy as np
import PIL


# Abstract base class for all transforms
class Transform(ABC):
    @abstractmethod
    def transform(self, data, input_index: Union[int, None], output_index: Union[int, None]):
        pass

    @abstractmethod
    def randomize(self):
        pass

    def transform_all(self, inputs, outputs, input_indices, output_indices):
        if input_indices is not None:
            for index in input_indices:
                inputs[index] = self.transform(inputs[index], index, None)
        if output_indices is not None:
            for index in output_indices:
                outputs[index] = self.transform(outputs[index], None, index)
        return inputs, outputs


class GammaIntensity(Transform):
    def __init__(self, min=0.25, max=1.75):
        self.min_gamma = min
        self.max_gamma = max

    def randomize(self):
        self.gamma = np.random.uniform(self.min_gamma, self.max_gamma)

    def transform(self, data, input_index, output_index):
        return np.clip(np.power(data, self.gamma), 0, 1)


class Rotation(Transform):
    def __init__(self, max_rotation_angle: float):
        self.max_rotation_angle = max_rotation_angle

    def randomize(self):
        self.angle = np.random.randint(-self.max_rotation_angle, self.max_rotation_angle)

    def transform(self, data, input_index, output_index):

        if input_index is None:
            # Rotate per channel
            # TODO improve this code
            is_3d = len(data.shape) == 4
            # Transform output
            if is_3d:
                data[:, :, :, 0] = np.ones(data.shape[:-1])  # Clear background
            else:
                data[:, :, 0] = np.ones(data.shape[:-1])  # Clear background
            for label in range(1, data.shape[-1]):
                if is_3d:
                    segmentation = scipy.ndimage.rotate(data[:, :, :, label], self.angle, axes=[1, 0], order=0,
                                                        reshape=False).reshape(data.shape[:-1])
                    data[:, :, :, label] = segmentation
                else:
                    segmentation = scipy.ndimage.rotate(data[:, :, label], self.angle, axes=[1, 0], order=0,
                                                        reshape=False).reshape(data.shape[:-1])
                    data[:, :, label] = segmentation

            # Remove segmentation from other labels
            for label in range(1, data.shape[-1]):
                for label2 in range(data.shape[-1]):
                    if label2 == label:
                        continue
                    if is_3d:
                        data[data[:, :, :, label] == 1, label2] = 0
                    else:
                        data[data[:, :, label] == 1, label2] = 0
        else:
            data = scipy.ndimage.rotate(data, self.angle, axes=[1, 0], order=1,
                                        reshape=False)  # Using order=2 here gives incorrect results

        return data


class Flip(Transform):
    def __init__(self, horizontal=True, vertical=False, depth=False):
        self.horizontal = horizontal
        self.vertical = vertical
        self.depth = depth

    def transform(self, data, input_index, output_index):
        if self.horizontal and self.flip_horizontal:
            data = np.flip(data, axis=1)
        if self.vertical and self.flip_vertical:
            data = np.flip(data, axis=0)
        if self.depth and self.flip_depth:
            data = np.flip(data, axis=2)

        return data

    def randomize(self):
        self.flip_horizontal = np.random.choice([False, True])
        self.flip_vertical = np.random.choice([False, True])
        self.flip_depth = np.random.choice([False, True])


class GaussianShadows(Transform):
    def __init__(self, nr_of_shadows: List[int] = [0, 1], strength_range=(0.75, 0.9), location_x_range=(-1.0, 1.0),
                 location_y_range=(-1.0, 1.0), sigma_x_range=(0.1, 0.4), sigma_y_range=(0.1, 0.4), symmetric: bool = False):
        self.strength_range = strength_range
        self.location_x_range = location_x_range
        self.location_y_range = location_y_range
        self.sigma_x_range = sigma_x_range
        self.sigma_y_range = sigma_y_range
        self.shadow_count_options = nr_of_shadows
        self.symmetric = symmetric

    def randomize(self):
        self.x_mu = np.random.uniform(self.location_x_range[0], self.location_x_range[1], 1).astype(np.float32)
        self.y_mu = np.random.uniform(self.location_y_range[0], self.location_y_range[1], 1).astype(np.float32)
        self.sigma_x = np.random.uniform(self.sigma_x_range[0], self.sigma_y_range[1], 1).astype(np.float32)
        if self.symmetric:
            self.sigma_y = self.sigma_x
        else:
            self.sigma_y = np.random.uniform(self.sigma_y_range[0], self.sigma_y_range[1], 1).astype(np.float32)
        self.strength = np.random.uniform(self.strength_range[0], self.strength_range[1], 1).astype(np.float32)
        self.nr_of_shadows = np.random.choice(self.shadow_count_options)

    def transform(self, data, input_index, output_index):
        size = data.shape
        x, y = np.meshgrid(np.linspace(-1, 1, size[1], dtype=np.float32), np.linspace(-1, 1, size[0], dtype=np.float32),
                           copy=False)
        for shadow in range(self.nr_of_shadows):
            g = 1.0 - self.strength * np.exp(
                -((x - self.x_mu) ** 2 / (2.0 * self.sigma_x ** 2) + (y - self.y_mu) ** 2 / (2.0 * self.sigma_y ** 2)),
                dtype=np.float32)

            data = data * np.reshape(g, size)

        return data

class JPEGCompression(Transform):
    def __init__(self, min_compression: int = 10, max_compression: int = 50, probability: float = 0.5):
        self.min_compression = min_compression
        self.max_compression = max_compression
        self.probability = probability

    def randomize(self):
        self.compression = int(np.random.randint(self.min_compression, self.max_compression, 1))
        self.compress_it = np.random.rand() <= self.probability

    def transform(self, data, input_index, output_index):
        if not self.compress_it:
            return data
        if data.shape[-1] == 1:
            mode = 'L'
            image = PIL.Image.fromarray((data[:, :, 0]*255).astype(np.uint8), mode)
        elif data.shape[-1] == 3:
            mode = 'RGB'
            image = PIL.Image.fromarray((data*255).astype(np.uint8), mode)
        else:
            raise ValueError('Unsupported nr of channels in JPEGCompression transform')

        with BytesIO() as f:
            image.save(f, format='JPEG', quality=100-self.compression)
            f.seek(0)
            image_jpeg = PIL.Image.open(f)
            result = np.asarray(image_jpeg).astype(np.float32) / 255.0
            result = result.copy()
        return result.reshape(data.shape)
