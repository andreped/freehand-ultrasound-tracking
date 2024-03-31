from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape
from tensorflow.python.keras.models import Model
#import tensorflow as tf


def convolution_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    for i in range(2):
        x = Conv2D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout2D(spatial_dropout)(x)
    return x


def encoder_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):

    x = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout)
    x = MaxPooling2D((2, 2))(x)

    return x


class VGGnet():
    def __init__(self, input_shape, nb_classes):
        if len(input_shape) != 3:
            raise ValueError('Input shape must have 3 dimensions')
        if nb_classes <= 1:
            raise ValueError('Classes must be > 1')
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.convolutions = None
        self.use_bn = True
        self.spatial_dropout = None
        self.dense_dropout = 0.5
        self.dense_size = 1024

    def set_dense_size(self, size):
        self.dense_size = size

    def set_dense_dropout(self, dropout):
        self.dense_dropout = dropout

    def set_spatial_dropout(self, dropout):
        self.spatial_dropout = dropout

    def set_convolutions(self, convolutions):
        self.convolutions = convolutions

    def get_depth(self):
        return len(self.convolutions)

    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer

        init_size = min(self.input_shape[0], self.input_shape[1])
        size = init_size

        convolutions = self.convolutions
        print(self.get_depth())
        if convolutions is None:
            raise ValueError('Number of convolutions in each layer must be given')

        i = 0
        for i in range(self.get_depth()):
        #while size > 4:
            x = encoder_block(x, convolutions[i], self.use_bn, self.spatial_dropout)
            #size /= 2
            #i+= 1

        x = convolution_block(x, convolutions[i], self.use_bn, self.spatial_dropout)

        x = Flatten(name="flatten")(x)
        x = Dense(self.dense_size, activation='relu')(x)
        if not self.dense_dropout == None:
            x = Dropout(self.dense_dropout)(x)
        #x = Dense(self.dense_size, activation='relu')(x)
        #x = Dropout(self.dense_dropout)(x)
        x = Dense(self.nb_classes, activation='linear')(x)

        return Model(inputs=input_layer, outputs=x)


