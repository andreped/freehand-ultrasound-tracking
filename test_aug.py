import os
from batch_generator import *
from keras_tools.network import Unet


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


#if __name__ == "__main__":

# use single GPU (first one)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# set name for model
name = "test"

# number of classes
num_classes = 7

# paths
path = "/home/andrep/anesthesia/"
data_path = path + "data/anesthesia_seg_12_07/"
save_model_path = path + "output/models/"
history_path = path + "output/history/"
datasets_path = path + "output/datasets/"

# Option 1: generate random sets by randomly assigning k patients to train, val and test set
locs = []
pats = []
for l in os.listdir(data_path):
    pats.append(l[:3])
pats = np.unique(pats)

k = 7
shuffle(pats)
test = pats[:k]
val = pats[k:int(2 * k)]
train = pats[int(2 * k):]

test_set = finder(test, data_path)
val_set = finder(val, data_path)
train_set = finder(train, data_path)

# define model
network = Unet(input_shape=(256, 256, 1), nb_classes=num_classes)
network.encoder_spatial_dropout = 0.2
network.decoder_spatial_dropout = 0.2
#network.set_convolutions([4, 8, 16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16, 8, 4])
network.set_convolutions([8, 16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16, 8])
model = network.create()

#
model.compile(
    optimizer='adadelta',
    loss=network.get_dice_loss()
)

# hyperpar
batch_size = 1
epochs = 4
imgs = 5

# augmentation
aug = {'flip': 1, 'gamma': [0.25, 1.75], 'rotate': 10, 'affine': [0.8, 1.0, 0.08, 0.1],\
             'shadow': [0.1*256, 0.9*256, 0.1*256, 0.9*256, 0.25, 0.8],
             'scale': [0.75, 1.5], 'shift': 30} # <--- added zoom aug

shuffle(train_set)
train_orig = train_set.copy()

while True:

    shuffle(train_orig)

    train_set = train_orig[:imgs]
    print(train_set)

    origs = train_set.copy()

    augs_list = []

    for curr_orig in train_set:

        train_set = [curr_orig]

        # generate sample
        for curr in train_set:

            print(curr)
            file = h5py.File(curr, 'r')
            data = np.array(file['data'], dtype=np.float32)
            gt = np.array(file['label'], dtype=np.uint8)
            file.close()

            orig = data[0,...,0]

            # define generators for sampling of data
            train_gen = batch_gen2(train_set, batch_size=batch_size, aug=aug, epochs=epochs)

            cnt = 0
            tmps = []

            print('----')
            augs = []

            for im, gt in train_gen:
                print(11)

                tmp = im[0,...,0]
                #tmp = gt[0,...,0]
                augs.append(tmp.copy())

                print(im.shape)

                cnt += 1

            augs_list.append(augs)



    fig, ax = plt.subplots(imgs, epochs+1, figsize=(10, 10))
    plt.tight_layout()

    print(len(origs))
    print()
    print(origs)
    print()
    print(len(augs_list))

    for i in range(len(origs)):
        f = h5py.File(origs[i], 'r')
        data = np.array(f['data'])
        gt = np.array(f['label'])
        f.close()

        orig = data[0, ..., 0]
        print(data.shape)
        ax[i,0].imshow(orig, cmap="gray")
        ax[i,0].set_title(origs[i].split('/')[-1].split('.h5')[0])

        print(i)
        print('-')

    print(len(augs_list))
    print(len(augs_list[0]))


    for i in range(len(augs_list)):
        for j in range(len(augs_list[0])):
            ax[i, j+1].imshow(augs_list[i][j], cmap="gray")
            print(i, j)

    for i in range(imgs):
        for j in range(epochs+1):
            #ax[i,j].axis('off')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])


    plt.show()
