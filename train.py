from math import ceil
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from batch_generator import *
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten,\
        BatchNormalization, SpatialDropout2D, Activation, GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.losses import logcosh
import h5py
from tensorflow.python.keras.optimizers import Adam, Adagrad
from datetime import date
from models_new import *


def initialize_weights(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)


def initialize_bias(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def abs_dx(y_true, y_pred):
    return tf.abs(y_true[0] - y_pred[0])

def abs_dy(y_true, y_pred):
    return tf.abs(y_true[1] - y_pred[1])

def abs_dz(y_true, y_pred):
    return tf.abs(y_true[2] - y_pred[2])

def abs_ax(y_true, y_pred):
    return tf.abs(y_true[3] - y_pred[3])

def abs_ay(y_true, y_pred):
    return tf.abs(y_true[4] - y_pred[4])

def abs_az(y_true, y_pred):
    return tf.abs(y_true[5] - y_pred[5])

def finder(test, path):
    sets = []
    for l in os.listdir(path):
        for t in test:
            if l.startswith(t):
                sets.append(path + l)
    return sets

def import_dataset(tmp, path, name_curr):
    file = h5py.File(path + 'dataset_' + name_curr + '.h5', 'r')
    tmp = np.array(file[tmp])
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    file.close()
    return tmp


def read_from_dir(train_dirs, data_path, sets=[]):
    tmp_set = []
    for d1 in train_dirs:
        p1 = data_path + d1 + "/"
        for d2 in os.listdir(p1):
            if d2 in sets or sets == []:  # <- [] means all
                p2 = p1 + d2 + "/"
                for d3 in os.listdir(p2):
                    p3 = p2 + d3
                    tmp_set.append(p3)
    return tmp_set


if __name__ == "__main__":

    # fix OOM problem in insecutive runs
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    ### Set Training params
    # use single GPU (first one)
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # choose whether to use pre-processed images, or pre-process on the fly
    img_flag = True

    # current date
    curr_date = "_".join(date.today().strftime("%d/%m/%Y").split("/")[:2])

    # number of classes
    nb_classes = 6

    # number of neighbours
    num_neighbours_train = 1 # <- using these types of aug apparently degraded performance (perhaps too large jumps?)
    num_neighbours_val = 1

    # set whether training or sandbox mode
    mode = "train" #"train" #"left_only" # train, check1, check2

    # model and data configs
    size_imgs = 256
    nb_images = 2
    nb_classes = 6

    # offset only
    offset_only = True
    if offset_only:
        nb_classes = 3

    # angles_only
    angles_only = False
    if angles_only:
        nb_classes = 3
    if angles_only and offset_only:
        print("OBS: Cannot have both offset_only and angles_only equal to TRUE!")
        exit()

    # xy offset only
    xy_only = False
    if xy_only:
        nb_classes = 2

    # training configs
    optim = "adam"
    lr = "1e-4" # "1e-4"

    # choos
    # e network
    net = 6 # 6
    conv_config = [16, 32, 64, 128] #[8,16,32,32,64,64,128]
    num_pools = len(conv_config)
    spatial_dropout = 0.1
    dense = 64 # 32 best?
    dropout = 0.25
    use_bn = False

    # hyperpar for training
    loss_fn = "mse" # <- mse/L2 og logcosh?
    batch_size = 128 #256/32 (too low, they used 500 in paper)  # <- large batch size important! (64 best?)
    epochs = 200

    # augmentation configs
    shadows = [0.1*size_imgs, 0.9*size_imgs, 0.1*size_imgs, 0.9*size_imgs, 0.25, 0.8]
    train_aug = {}
    train_aug = {'gamma':[0.8,1.2]}
    #train_aug = {'gamma':[0.8, 1.2]}
    #train_aug = {'gamma':[0.8, 1.2],'shadow':shadows} # this aug combination only seem to degrade performance
    val_aug = {}
    #val_aug = {'flipdir':1,'gamma':[0.8,1.2]}

    # data set
    # which mode data was created with
    modes = "10_only" #"5_only", "normal"

    # set name for model
    name = curr_date + "_test_bs_" + str(batch_size) + "_loss_" + loss_fn + "_augs_" + \
           str(train_aug)[1:-1].replace("'", "") + "_" + str(val_aug)[1:-1].replace("'", "") + "_data_both_net_" \
           + str(net) + "_conv_config_" + str(conv_config)[1:-1].replace(" ", "") + "_spatial_dropout" \
           + str(spatial_dropout) + "_dense_" + str(dense) + "_dropout_" + str(dropout) \
           + "_bn_" + str(use_bn).lower() + "_opt_" + optim +"_lr_" + lr + "_img_" + str(img_flag).lower() + \
           "_mode_" + str(mode) + "_data_set_" + modes
    print("name of training")
    print(name)

    # paths
    path = "/home/andrep/workspace/freehand_tracking/"
    data_path = path + "data/anesthesia_tracking_23_08_num_neighbours_" + str(num_neighbours_train) + "_images_" + str(img_flag) + "_mode_" + modes+ "/"
    save_model_path = path + "output/models/"
    history_path = path + "output/history/"
    datasets_path = path + "output/datasets/"
    data_orig_path = path + "data/anesthesia_tracking_23_08_num_neighbours_" + str(num_neighbours_val) + "_images_" + str(img_flag) + "_mode_" + modes + "/"

    if mode == "train":
        # fixed split
        train_dirs = ["0_left", "0_right", "2_left", "2_right", "3_left", "3_right", "4_left", "4_right"]
        #val_dirs = ["5_left", "5_right"]
        #test_dirs = ["6_left", "6_right"]
        #train_dirs = ["0_left", "0_right", "2_left", "2_right", "4_left", "4_right"]
        val_dirs = ["5_left", "5_right", "6_left", "6_right"]
        #val_dirs = val_dirs + test_dirs
        test_dirs = val_dirs.copy()

        # sets=["linear", "linear-back-and-forth", "natural", "crazy"])
        sets = ["linear-back-and-forth"]
        train_set = read_from_dir(train_dirs, data_path, sets=sets)
        val_set = read_from_dir(val_dirs, data_orig_path, sets=sets)
        test_set = read_from_dir(test_dirs, data_orig_path, sets=sets)

    # sanity checks
    # check 1 (It works!)
    if mode == "check1":
        sets = "linear-back-and-forth"
        tmp = os.listdir(data_path + "5_left" + "/" + sets)
        tmps = []
        for t in tmp:
            tmps.append(data_path + "5_left" + "/" + sets + "/" + t)
        train_set = tmps[int(len(tmps)/2):]
        val_set = tmps[:int(len(tmps)/2)]
        test_set = val_set.copy()

    # check 2 (Fails!)
    if mode == "check2":
        sets = "linear-back-and-forth"
        pat = "0"
        tmp = os.listdir(data_path + pat + "_left" + "/" + sets)
        tmps = []
        for t in tmp:
            tmps.append(data_path + pat + "_left" + "/" + sets + "/" + t)
        train_set = tmps.copy()

        tmp = os.listdir(data_orig_path + pat + "_right" + "/" + sets)
        tmps = []
        for t in tmp:
            tmps.append(data_orig_path + pat + "_right" + "/" + sets + "/" + t)
        val_set = tmps.copy()
        test_set = val_set.copy()

    # check 3
    if mode == "left_only":
        sets = "linear-back-and-forth"
        # fixed split
        #train_dirs = ["0_left", "1_left", "2_left", "5_left", "6_left"]
        #val_dirs = ["3_left"]
        #test_dirs = ["4_left"]
        train_dirs = ["0_left", "2_left", "4_left"]
        val_dirs = ["5_left"]
        #val_dirs = val_dirs + test_dirs
        test_dirs = val_dirs.copy()

        # sets=["linear", "linear-back-and-forth", "natural", "crazy"])
        train_set = read_from_dir(train_dirs, data_path, sets=[sets])
        val_set = read_from_dir(val_dirs, data_path, sets=[sets])
        test_set = read_from_dir(test_dirs, data_path, sets=[sets])


    # save generated data sets
    f = h5py.File((datasets_path + 'dataset_' + name + '.h5'), 'w')
    f.create_dataset("test", data=np.array(test_set).astype('S200'), compression="gzip", compression_opts=4)
    f.create_dataset("val", data=np.array(val_set).astype('S200'), compression="gzip", compression_opts=4)
    f.create_dataset("train", data=np.array(train_set).astype('S200'), compression="gzip", compression_opts=4)
    f.close()


    ## define model
    if net == 1:
        model = Sequential()
        model.add(Conv2D(64, (5,5), input_shape=(int(size_imgs), int(size_imgs), nb_images), strides=2))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5,5), strides=2))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))

        model.add(Conv2D(64, (3,3), strides=2))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3,3), strides=2))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(nb_classes, activation='linear'))

    elif net == 2:
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(int(size_imgs), int(size_imgs), nb_images), strides=2))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), strides=2))
        model.add(Activation('relu'))
        model.add(SpatialDropout2D(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(32, (3, 3), strides=2))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), strides=2))
        model.add(Activation('relu'))
        model.add(SpatialDropout2D(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='linear'))

    elif net == 3:  # <- without augs
        model = Sequential()
        model.add(Conv2D(16, (5, 5), input_shape=(int(size_imgs), int(size_imgs), nb_images)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (5, 5)))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), strides=2))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), strides=2))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(32, (3, 3), strides=2))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), strides=2))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Flatten())
        model.add(Dense(64, activation='relu')) # 32, without any aug
        model.add(Dropout(0.25))
        model.add(Dense(nb_classes, activation='linear'))

    elif net == 4:  # <- with augs
        # bottom model
        model = Sequential()
        for i in range(len(conv_config)):
            if i == 0:
                model.add(Conv2D(conv_config[i], (3, 3), input_shape=(int(size_imgs), int(size_imgs), nb_images)))
            else:
                model.add(Conv2D(conv_config[i], (3, 3)))
            model.add(Activation('relu'))
            model.add(Conv2D(conv_config[i], (3, 3)))
            model.add(Activation('relu'))
            model.add(SpatialDropout2D(0.2))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        # top model
        model.add(Flatten())
        model.add(Dense(dense, activation='relu')) # 32, without any aug
        if not dropout == None:
            model.add(Dropout(dropout))
        model.add(Dense(nb_classes, activation='linear'))

    elif net == 5:
        # bottom model
        model = Sequential()
        for i in range(len(conv_config)):
            for j in range(2):
                if i == 0 and j == 0:
                    model.add(Conv2D(conv_config[i], (3, 3), input_shape=(int(size_imgs), int(size_imgs), nb_images)))
                else:
                    model.add(Conv2D(conv_config[i], (3, 3)))
                if use_bn:
                    model.add(BatchNormalization())
                model.add(Activation('relu'))
                if not spatial_dropout == None:
                    model.add(SpatialDropout2D(spatial_dropout))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        # top model
        model.add(Flatten())
        model.add(Dense(dense, activation='relu'))  # 32, without any aug
        if not dropout == None:
            model.add(Dropout(dropout))
        model.add(Dense(nb_classes, activation='linear'))

    elif net == 6:
        # bottom model
        model = Sequential()
        for i in range(len(conv_config)):
            for j in range(2):
                if i == 0 and j == 0:
                    model.add(Conv2D(conv_config[i], (3, 3), input_shape=(int(size_imgs), int(size_imgs), nb_images)))
                else:
                    model.add(Conv2D(conv_config[i], (3, 3)))
                if use_bn:
                    model.add(BatchNormalization())
                model.add(Activation('relu'))
            if not dropout == None:
                model.add(Dropout(dropout))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        # top model
        model.add(Flatten())
        model.add(Dense(dense, activation='relu'))  # 32, without any aug
        if not dropout == None:
            model.add(Dropout(dropout))
        #model.add(Dense(dense, activation='relu'))  # 32, without any aug
        #if not dropout == None:
        #    model.add(Dropout(dropout))
        model.add(Dense(nb_classes, activation='linear'))

    elif net == 7:
        # bottom model
        model = Sequential()
        model.add(Conv2D(8, (10, 10), strides=2, padding='same', input_shape=(int(size_imgs), int(size_imgs), nb_images)))
        model.add(Activation('relu'))
        model.add(Conv2D(8, (10, 10), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (7, 7), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (7, 7), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        '''
        model.add(Conv2D(32, (5, 5), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=2, padding='same'))
        model.add(Activation('relu'))
        '''
        model.add(GlobalAveragePooling2D())

        # top model
        model.add(Flatten())
        model.add(Dense(dense, activation='relu'))  # 32, without any aug
        if not dropout == None:
            model.add(Dropout(dropout))
        model.add(Dense(nb_classes, activation='linear'))

    elif net == "FCN":
        model = Sequential()
        for i in range(len(conv_config)):
            for j in range(2):
                if i == 0 and j == 0:
                    model.add(Conv2D(conv_config[i], (3, 3), input_shape=(int(size_imgs), int(size_imgs), nb_images)))
                else:
                    model.add(Conv2D(conv_config[i], (3, 3)))
                if use_bn:
                    model.add(BatchNormalization())
                model.add(Activation('relu'))
            if not dropout == None:
                model.add(Dropout(dropout))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(dense, (3, 3), activation='relu'))
        if not dropout == None:
            model.add(Dropout(dropout))
        model.add(Conv2D(nb_classes, (1, 1), activation='linear'))

    elif net == "siamese":
        from smistad.network import *
        from tensorflow.python.keras.layers import merge, concatenate
        import tensorflow as tf
        from tensorflow.python.keras.regularizers import l2

        input_shape = (int(size_imgs), int(size_imgs), 1)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        model = Sequential()
        model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
        model.add(Flatten())
        model.add(Dense(4096, activation='sigmoid',
                        kernel_regularizer=l2(1e-3)))

        encoded_l = model(left_input)
        encoded_r = model(right_input)
        L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        prediction = Dense(nb_classes, activation='linear')(L1_distance)
        model = Model(inputs=[left_input, right_input], outputs=prediction)


    elif net == 10:
        network = VGGnet(input_shape=(int(size_imgs), int(size_imgs), 2), nb_classes=nb_classes)
        # network.set_convolutions([16,32,64,128,256,512])  # <- more features (512x512) => too complex...
        network.set_convolutions(conv_config)  # <- even smaller net
        network.set_dense_size(dense)  # 100
        network.set_dense_dropout(dropout)  # default 0.5
        network.use_bn = use_bn
        network.set_spatial_dropout(spatial_dropout)  # 0.2
        model = network.create()

    elif net == 11:
        from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, \
            SpatialDropout2D, \
            ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
            TimeDistributed, Concatenate, Lambda, Reshape
        from tensorflow.python.keras.models import Model, Sequential
        from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
        from tensorflow.python.keras.applications.inception_v3 import InceptionV3

        input_tensor = Input(shape=(int(size_imgs), int(size_imgs), 2))
        base_model = InceptionV3(include_top=False, weights='imagenet',
                                 input_tensor=input_tensor, pooling='max')

        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        # top_model.add(Dense(128, activation='relu'))  # 512
        # top_model.add(Dropout(rate=0.5))
        top_model.add(Dense(64, activation='relu'))  # 32 best
        top_model.add(Dropout(rate=0.5))
        top_model.add(Dense(nb_classes, activation='linear'))

        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))


    print(model.summary())

    def myAdam(lr):
        Adam(lr)
    # getattr(myAdam(float(lr)), "name")

    # optimization configs
    metrics = ['mse', abs_dx, abs_dy, abs_dz, abs_ax, abs_ay, abs_az]
    if offset_only:
        metrics = ['mse', abs_dx, abs_dy, abs_dz]
    if angles_only:
        metrics = ['mse', abs_ax, abs_ay, abs_az]
    if xy_only:
        metrics = ['mse', abs_ax, abs_ay]
    model.compile(
        optimizer=Adam(float(lr)),
        loss=loss_fn,
        metrics=metrics
    )

    train_gen = batch_gen2(train_set, batch_size, aug=train_aug, num_classes=nb_classes, epochs=epochs, img_flag=img_flag, offset_only=offset_only, angles_only=angles_only, xy_only=xy_only)
    val_gen = batch_gen2(val_set, batch_size, aug=val_aug, num_classes=nb_classes, epochs=epochs, img_flag=img_flag, offset_only=offset_only, angles_only=angles_only, xy_only=xy_only)

    train_length = len(train_set)
    val_length = len(val_set)

    # Reduce learning rate on plateau
    lr_reduce_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=50
    )

    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=1000,
    )

    save_best = ModelCheckpoint(
        save_model_path + "model_" + name + ".h5",
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1
    )

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.losses.append(['loss', 'val_loss', 'logcosh', 'val_logcosh', 'mse', 'log_mse'])

        def on_epoch_end(self, batch, logs={}):
            self.losses.append([logs.get('loss'), logs.get('val_loss'),
                                logs.get('logcosh'), logs.get('val_logcosh'),
                                logs.get('mse'), logs.get('val_mse')])
            # save history:
            f = h5py.File(history_path + 'history_' + name + '.h5', 'w')
            f.create_dataset("history", data=np.array(self.losses).astype('|S9'), compression="gzip", compression_opts=4)
            f.close()

    history_log = LossHistory()

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=int(ceil(train_length / batch_size)),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=int(ceil(val_length / batch_size)),
        callbacks=[save_best, history_log],
        use_multiprocessing=False,
        workers=1
    )
