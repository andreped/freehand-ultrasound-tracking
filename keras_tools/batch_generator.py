from __future__ import absolute_import
from __future__ import print_function
import sys
from typing import List, Union
from .data_transform import Transform
try:
    from keras.preprocessing.image import Iterator
except:
    from tensorflow.python.keras.preprocessing.image import Iterator
import numpy as np
from six.moves import range
import h5py
import time


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


class BatchIterator(Iterator):
    def __init__(self, generator, file_list, input_shape, output_shape, batch_size, shuffle, all_files_in_batch):
        self.file_list = file_list
        self.generator = generator
        self.input_shape = input_shape
        self.nr_of_inputs = len(input_shape)
        self.output_shape = output_shape
        self.nr_of_outputs = len(output_shape)
        self.all_files_in_batch = all_files_in_batch
        self.preload_to_memory = False
        self.file_cache = {}
        self.max_cache_size = 10*1024
        self.verbose = False
        if self.preload_to_memory:
            for filename, file_index in self.file_list:
                file = h5py.File(filename, 'r')
                inputs = {}
                for name, data in file['input'].items():
                    inputs[name] = np.copy(data)
                self.file_cache[filename] = {'input': inputs, 'output': np.copy(file['output'])}
                file.close()
                if get_size(self.file_cache) / (1024*1024) >= self.max_cache_size:
                    print('File cache has reached limit of', self.max_cache_size, 'MBs')
                    break
        epoch_size = len(file_list)
        if all_files_in_batch:
            epoch_size = len(file_list) * 10
        super(BatchIterator, self).__init__(epoch_size, batch_size, shuffle, None)

    def _get_sample(self, index):
        filename, file_index = self.file_list[index]
        if filename in self.file_cache:
            file = self.file_cache[filename]
        else:
            file = h5py.File(filename, 'r')
        inputs = []
        outputs = []
        for name, data in file['input'].items():
            inputs.append(data[file_index, :])
        for name, data in file['output'].items():
            outputs.append(data[file_index, :])
        #outputs.append(file['output'][file_index, :]) # TODO fix
        if filename not in self.file_cache:
            file.close()
        return inputs, outputs

    def _get_random_sample_in_file(self, file_index):
        filename = self.file_list[file_index]
        file = h5py.File(filename, 'r')
        x = file['output']
        sample = np.random.randint(0, x.shape[0])
        #print('Sampling image', sample, 'from file', filename)
        inputs = []
        outputs = []
        for name, data in file['input'].items():
            inputs.append(data[sample, :])
        for name, data in file['output'].items():
            outputs.append(data[file_index, :])
        #outputs.append(file['output'][sample, :]) # TODO FIX output
        file.close()
        return inputs, outputs

    def next(self):

        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        start_batch = time.time()
        batches_x = []
        batches_y = []
        for input_index in range(self.nr_of_inputs):
            batches_x.append(np.zeros(tuple([len(index_array)] + list(self.input_shape[input_index]))))
        for output_index in range(self.nr_of_outputs):
            batches_y.append(np.zeros(tuple([len(index_array)] + list(self.output_shape[output_index]))))

        timings_sampling = np.zeros((len(index_array,)))
        timings_transform = np.zeros((len(index_array,)))
        for batch_index, sample_index in enumerate(index_array):
            # Have to copy here in order to not modify original data
            start = time.time()
            if self.all_files_in_batch:
                input, output = self._get_random_sample_in_file(batch_index)
            else:
                input, output = self._get_sample(sample_index)
            timings_sampling[batch_index] = time.time() - start
            start = time.time()
            input, output = self.generator.transform(input, output)
            timings_transform[batch_index] = time.time() - start

            #print('inputs', self.nr_of_inputs, len(input))
            for input_index in range(self.nr_of_inputs):
                batches_x[input_index][batch_index] = input[input_index]
            for output_index in range(self.nr_of_outputs):
                batches_y[output_index][batch_index] = output[output_index]

        elapsed = time.time() - start_batch
        if self.verbose:
            print('Time to prepare batch:', round(elapsed,3), 'seconds')
            print('Sampling mean:', round(timings_sampling.mean(), 3), 'seconds')
            print('Transform mean:', round(timings_transform.mean(), 3), 'seconds')

        return batches_x, batches_y


CLASSIFICATION = 'classification'
SEGMENTATION = 'segmentation'


class BatchGenerator():
    def __init__(self, filelist, all_files_in_batch=False):
        self.methods = []
        self.args = []
        self.crop_width_to = None
        self.image_list = []
        self.input_shape = []
        self.output_shape = []
        self.all_files_in_batch = all_files_in_batch
        self.transforms = []

        if all_files_in_batch:
            file = h5py.File(filelist[0], 'r')
            for name, data in file['input'].items():
                self.input_shape.append(data.shape[1:])
            for name, data in file['output'].items():
                self.output_shape.append(data.shape[1:])
            # TODO fix
            self.output_shape.append(file['output'].shape[1:])
            file.close()
            self.image_list = filelist
            return

        # Go through filelist
        first = True
        for filename in filelist:
            samples = None
            # Open file to see how many samples it has
            file = h5py.File(filename, 'r')
            for name, data in file['input'].items():
                if first:
                    self.input_shape.append(data.shape[1:])
                samples = data.shape[0]
            # TODO fix
            for name, data in file['output'].items():
                if first:
                    self.output_shape.append(data.shape[1:])
                if samples != data.shape[0]:
                    raise ValueError()
            #self.output_shape.append(file['output'].shape[1:])
            if len(self.output_shape) == 1:
                self.problem_type = CLASSIFICATION
            else:
                self.problem_type = SEGMENTATION

            file.close()
            if samples is None:
                raise ValueError()
            # Append a tuple to image_list for each image consisting of filename and index
            for i in range(samples):
                self.image_list.append((filename, i))
            first = False

        print('Image generator with', len(self.image_list), ' image samples created')

    def flow(self, batch_size, shuffle=True):

        return BatchIterator(self, self.image_list, self.input_shape, self.output_shape, batch_size, shuffle, self.all_files_in_batch)

    def transform(self, inputs, outputs):
        #input = input.astype(np.float32) # TODO
        #output = output.astype(np.float32)
        for input_indices, output_indices, transform in self.transforms:
            transform.randomize()
            inputs, outputs = transform.transform_all(inputs, outputs, input_indices, output_indices)
        return inputs, outputs

    def add_transform(self, input_indices: Union[int, List[int], None], output_indices: Union[int, List[int], None], transform: Transform):
        if type(input_indices) is int:
            input_indices = [input_indices]
        if type(output_indices) is int:
            output_indices = [output_indices]

        self.transforms.append((
            input_indices,
            output_indices,
            transform
        ))

    def get_size(self):
        if self.all_files_in_batch:
            return 10*len(self.image_list)
        else:
            return len(self.image_list)
