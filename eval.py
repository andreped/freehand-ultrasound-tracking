import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
import cv2
from tqdm import tqdm
from numpy.random import shuffle
import tensorflow as tf
from smistad.smistad_network import Unet  # :(
import os
from erik_code.post_process import post_process, post_process2, post_process3
from skimage.measure import label, regionprops

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    precision_recall_fscore_support
from keras_tools.metaimage import *
from matplotlib.colors import Normalize

# surpress scientific notation
np.set_printoptions(suppress=True)


def resizer(data, out_dim, gt=True):
    orig_dim = data.shape
    scale = out_dim[0] / data.shape[1]
    if not gt:
        #data = transform.rescale(data, scale=scale, preserve_range=True, order=1, multichannel=False)  # This also transforms image to be between 0 and 1, therefore preserve_range=True
        data = cv2.resize(data.astype(np.uint8), (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        #data = transform.rescale(data, scale=scale, preserve_range=True, order=0, multichannel=False)
        data = cv2.resize(data.astype(np.uint8), (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(np.float32)

    if data.shape[0] > orig_dim[0]:
        # cut image
        data = data[:orig_dim[0], :]
    elif data.shape[0] < orig_dim[0]:
        tmp = np.zeros(orig_dim, dtype=np.float32)
        tmp[:data.shape[0], :out_dim[0]] = data
        data = tmp[:out_dim[0], :out_dim[1]]
    return data


class PiecewiseNormalize(Normalize):
    def __init__(self, xvalues, cvalues):
        self.xvalues = xvalues
        self.cvalues = cvalues

        Normalize.__init__(self)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        if self.xvalues is not None:
            x, y = self.xvalues, self.cvalues
            return np.ma.masked_array(np.interp(value, x, y))
        else:
            return Normalize.__call__(self, value, clip)


def import_set(tmp, num=None, filter=False):
    f = h5py.File(datasets_path + 'dataset_' + name + '.h5', 'r')
    tmp = np.array(f[tmp])
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    #shuffle(tmp)
    if filter:
        tmp = remove_copies(tmp)
    if num != None:
        tmp = tmp[:num]
    f.close()
    return tmp

def DSC(target, output):
    dice = 0
    #epsilon = 1e-10
    for object in range(1, output.shape[-1]):
        output1 = output[..., object]
        target1 = target[..., object]
        intersection = np.sum(output1 * target1)
        union = np.sum(output1 * output1) + np.sum(target1 * target1)
        dice += 2. * intersection / union
    dice /= (output.shape[-1] - 1)
    return dice

def DSC_simple(target, output):
    #epsilon = 1e-10
    intersection = np.sum(output * target)
    union = np.sum(output * output) + np.sum(target * target)
    dice = 2. * intersection / union
    return dice

def oneHoter(tmp, labels):
    res = np.zeros(tmp.shape + (labels, ))
    for l in range(labels):
        tmp2 = np.zeros_like(tmp)
        tmp2[tmp == l] = 1
        res[..., l] = tmp2
    return res


# set seed
np.random.seed(42)

# choose whether to run on GPU or not, and for which GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# set name for model
name = "27_08_test_bs_500"
name = "27_08_test_bs_8_loss_mse"
name = "27_08_test_bs_8_loss_mse_linear_only"
name = "27_08_test_bs_8_loss_mse_linear_only_augs_flip_data_left_split"

# number of classes
nb_classes = 6

# paths
path = "/home/andrep/workspace/freehand_tracking/"
data_path = path + "data/anesthesia_tracking_23_08/"
save_model_path = path + "output/models/"
history_path = path + "output/history/"
datasets_path = path + "output/datasets/"

# load test_data_set
test_set = import_set('test')
val_set = import_set('val')
train_set = import_set('train')

# choose set
sets = train_set.copy()
sets = np.array(sets)

# shuffle set
shuffle(sets)

# load trained model
model = load_model(save_model_path + "model_" + name + '.h5', compile=False)

input_shape=(1, 256, 256, 2)

diffs = []

counter = 0
for path in tqdm(np.array(sets)):
    # load data
    file = h5py.File(path, 'r')
    imgs = np.array(file['input'])
    gt = np.array(file['output']).astype(np.float32)
    file.close()

    # load and preprocess images
    data = np.zeros(input_shape[1:])
    for i, img in enumerate(imgs):
        img = img.decode("UTF-8")
        data_object = MetaImage(img.split(".")[0] + ".mhd")
        data_tmp = np.asarray(data_object.get_image()).astype(np.float32)
        data[..., i] = resizer(data_tmp, out_dim=(256, 256), gt=False)
    data = data / 255 # normalize

    # predict
    pred = model.predict(np.expand_dims(data, axis=0))[0]

    diffs.append(pred - gt)

    print('---')
    print(pred)
    print(gt)
    print()

print("Results: ")
print(np.abs(np.mean(diffs, axis=0)))
print(np.abs(np.std(diffs, axis=0)))
