

import numpy as np 
import h5py
from prettytable import PrettyTable
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

#name = "15_07_multi_seg_7_classes_spat_drop_0.2_256_deepest"
#name = "15_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_8"
#name = "15_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_16"
name = "15_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_4"
name = "15_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_8_zoom_aug"
name = "15_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_8_zoom_aug_adam"
name = "16_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_8_zoom_aug_adam"
name = "16_07_multi_seg_7_classes_spat_drop_0.2_deepest_bs_6_zoom_aug_adam_decay_stale_50_2000_epochs"
name = "17_07_multi_seg_7_classes_spat_drop_0.1_deepest_bs_6_zoom_aug_adam_decay_stale_50_2000_epochs"
name = "17_07_multi_seg_7_classes_spat_drop_0.1_deepest_bs_32_zoom_aug_adam_decay_stale_50_2000_epochs"


path = "/home/andrep/anesthesia/output/history/history_" + name + ".h5"
f = h5py.File(path, 'r')
data = np.array(f["history"])
f.close()

data = data[:,(0,1)]
#data[1:,2] = np.array([0.7285, 0.8807, 0.9087, 0.9185, 0.9332, 0.9417, 0.9533, 0.9595, 0.9689, 0.9735,
#	0.9758, 0.9744, 0.9810, 0.9807, 0.9797, 0.9856, 0.9260, 0.9718, 0.9818, 0.9818, 0.9817])
#data[1:,3] = np.array([0.4598, 0.4183, 0.5018, 0.5988, 0.6231, 0.6426, 0.5194, 0.6338, 0.7472, 0.8083,
#	0.6081, 0.7060, 0.6635, 0.5885, 0.6780, 0.8986, 0.7969, 0.6906, 0.6759, 0.7750, 0.8123])

#metrics = [data[0][x].decode("UTF-8") for x in range(len(data[0]))]

# remove unicode b from all eleiments
data = np.reshape(np.array([data[x,y].decode("UTF-8") for x in range(data.shape[0]) for y in range(data.shape[1])]), data.shape)

# split metrics from results
metrics = data[0,:]
data = np.round(data[1:,:].astype(np.float32), 8)
data = np.array(data).astype(float)

# make table of results
x = PrettyTable()
x.title = 'Training history'
metrics = metrics.tolist()
metrics.insert(0, "epochs")
x.field_names = metrics
epochs = np.array(range(data.shape[0]))
tmp = np.zeros(data.shape[1]+1, dtype=object)
for i in range(data.shape[0]):
    tmp[0] = i+1
    a = np.round(data[i,:], 5)
    tmp[1:] = a
    x.add_row(tmp)

# set epoch-column to ints
# ...

print(x)

# quick summary of lowest val_loss and for which epoch
print('Lowest val_loss observed: ')
print(np.amin(data[:,1]))
print('At epoch: ')
print(np.argmin(data[:,1])+1)

epochs = np.array(range(1, data.shape[0]+1))

fig, ax = plt.subplots(1,1, figsize=(14,8))
plt.tight_layout()
ax.plot(epochs, data[:,0])
ax.plot(epochs, data[:,1])
ax.plot([np.argmin(data[:,1])+1, np.argmin(data[:,1])+1], [np.amin(data), np.amax(data)], 'k:')
ax.set_ylim([np.amin(data), np.amax(data)])
ax.set_xlim([min(epochs), max(epochs)])
ax.set_xlabel('epochs')
ax.set_ylabel('DSC')
ax.legend(['train', 'val'], loc='best')
ax.set_title("Lowest val_loss observed: " + str(np.round(np.amin(data[:,1]), 4)) + ", at epoch: " + str(np.argmin(data[:,1])+1))
plt.grid(True)

plt.show()
