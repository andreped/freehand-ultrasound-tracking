import zlib
import numpy as np
import PIL, PIL.Image
import os


def tuple_to_string(tuple):
    string = ''
    for i in range(len(tuple)):
        string += str(tuple[i]) + ' '
    return string[:len(string)-1]


# Class for loading and storing image data in metaimage format (.mhd)
class MetaImage:
    def __init__(self, filename=None, data=None):
        self.attributes = {}
        self.attributes['ElementSpacing'] = [1, 1, 1]
        if filename is not None:
            self.read(filename)
        else:
            self.ndims = len(data.shape)
            self.data = data

            # Switch so that we get width first
            if self.ndims == 2:
                self.dim_size = (data.shape[1], data.shape[0])
            else:
                self.dim_size = (data.shape[1], data.shape[0], data.shape[2])

    def read(self, filename):
        if not os.path.isfile(filename):
            raise Exception('File ' + filename + ' does not exist')

        base_path = os.path.dirname(filename) + '/'
        # Parse file
        with open(filename, 'r') as f:
            for line in f:
                if line.strip() == '': # Skip empty line
                    continue
                parts = line.split('=')
                if len(parts) != 2:
                    raise Exception('Unable to parse metaimage file')
                self.attributes[parts[0].strip()] = parts[1].strip()

        if 'CompressedData' in self.attributes and self.attributes['CompressedData'] == 'True':
            # Read compressed raw file (.zraw)
            with open(os.path.join(base_path, self.attributes['ElementDataFile']), 'rb') as raw_file:
                raw_data_compressed = raw_file.read()
                raw_data_uncompressed = zlib.decompress(raw_data_compressed)
                self.data = np.fromstring(raw_data_uncompressed, dtype=np.uint8)
        else:
            # Read uncompressed raw file (.raw)
            self.data = np.fromfile(os.path.join(base_path, self.attributes['ElementDataFile']), dtype=np.uint8)

        dims = self.attributes['DimSize'].split(' ')
        new_dims = []
        for dim in dims:
            if len(dim) > 0:
                new_dims.append(dim)
        dims = new_dims
        self.ndims = int(self.attributes['NDims'])
        if self.ndims != len(dims):
            raise Exception('NDims and DimSize not matching')
        if len(dims) == 2:
            self.dim_size = (int(dims[0]), int(dims[1]))
            self.data = self.data.reshape((self.dim_size[1], self.dim_size[0]))
        elif len(dims) == 3:
            self.dim_size = (int(dims[0]), int(dims[1]), int(dims[2]))
            # If last dimension is 1, remove it and make image 2D
            if self.dim_size[2] == 1:
                self.dim_size = self.dim_size[:2]
                self.ndims = 2
                self.data = self.data.reshape((self.dim_size[1], self.dim_size[0]))
            else:
                self.data = self.data.reshape((self.dim_size[1], self.dim_size[0], self.dim_size[2]))



    def get_size(self):
        return self.dim_size

    def get_pixel_data(self):
        return self.data

    def get_image(self):
        pil_image = PIL.Image.new('L', self.get_size(), color='white')
        pil_image.putdata(self.data.flatten())

        return pil_image

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_spacing(self, spacing):
        if len(spacing) != 2 and len(spacing) != 3:
            raise ValueError('Spacing must have 2 or 3 components')
        self.attributes['ElementSpacing'] = spacing

    def get_spacing(self):
        if isinstance(self.attributes['ElementSpacing'], str):
            return [float(x) for x in self.attributes['ElementSpacing'].split()]
        else:
            return self.attributes['ElementSpacing']

    def get_metaimage_type(self):
        np_type = self.data.dtype
        if np_type == np.float32:
            return 'MET_FLOAT'
        elif np_type == np.uint8:
            return 'MET_UCHAR'
        elif np_type == np.int8:
            return 'MET_CHAR'
        elif np_type == np.uint16:
            return 'MET_USHORT'
        elif np_type == np.int16:
            return 'MET_SHORT'
        elif np_type == np.uint32:
            return 'MET_UINT'
        elif np_type == np.int32:
            return 'MET_INT'
        else:
            raise ValueError('Unknown numpy type')

    def write(self, filename):
        base_path = os.path.dirname(filename) + '/'
        filename = os.path.basename(filename)
        raw_filename = filename[:filename.rfind('.')] + '.raw'
        # Write meta image file
        with open(base_path + filename, 'w') as f:
            f.write('NDims = ' + str(self.ndims) + '\n')
            f.write('DimSize = ' + tuple_to_string(self.dim_size) + '\n')
            f.write('ElementType = ' + self.get_metaimage_type() + '\n')
            f.write('ElementSpacing = ' + tuple_to_string(self.attributes['ElementSpacing']) + '\n')
            for key, value in self.attributes.items():
                if key not in ['NDims', 'DimSize', 'ElementType', 'ElementDataFile', 'CompressedData', 'CompressedDataSize', 'ElementSpacing']:
                    f.write(key + ' = ' + value + '\n')
            f.write('ElementDataFile = ' + raw_filename + '\n')

        # TODO compressed storage

        # Write raw file
        with open(base_path + raw_filename, 'wb') as f:
            f.write(np.vstack(self.data).tobytes())