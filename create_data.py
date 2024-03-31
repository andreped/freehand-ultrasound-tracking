from keras_tools.metaimage import *
import h5py
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from eulerangles import mat2euler, euler2mat
import shutil
import numpy as np
import multiprocessing as mp
np.set_printoptions(suppress=True)
from math import ceil


# Computes entropy of label distribution
def entropy2(labels, base=None):
    from math import log, e
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent

# converts reflection matrix to rotation matrix. Reflection matrix has det = -1, while rotation matrix has det = +1
def ref2rot(mat):
    mat[:, 2] = np.cross(mat[:, 0], mat[:, 1])
    return mat

def oneHoter(tmp, labels):
    res = np.zeros(tmp.shape + (labels, ))
    for l in range(labels):
        tmp2 = np.zeros_like(tmp)
        tmp2[tmp == l] = 1
        res[..., l] = tmp2
    return res

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


def func(pat):
    print(pat)
    curr1 = data_path + pat
    end_path1 = end_path + pat + "/"
    if not os.path.exists(end_path1):
        os.mkdir(end_path1)
    for scan in os.listdir(curr1): # type of scan: linear, back&forth linear, natural, crazy
        end_path2 = end_path1 + scan + "/"
        if not os.path.exists(end_path2):
            os.mkdir(end_path2)

        # get ordered paths to images
        scan_tmp = []
        curr2 = curr1 + "/" + scan
        for img in os.listdir(curr2):
            if img.endswith("raw"):
                curr3 = curr2 + "/" + img
                scan_tmp.append(curr3)
        imgs = []
        for s in scan_tmp:
            imgs.append([int(s.split("/")[-1].split(".")[0].split("_")[-1]), s])
        imgs = np.array(imgs)
        tmp = imgs[:, 0].astype(int)
        order = sorted(range(len(tmp)), key=lambda k: tmp[k])
        new_imgs = imgs[order, 1]

        # test
        if "_only" in mode:
            k = int(mode.split("_only")[0])
            new = []
            for i in range(len(new_imgs)-k-1):
                new.append([new_imgs[i], new_imgs[i + k]])

        # creates non-adjacent pairs (jumps over nearest)
        else:
            new = []
            for i in range(len(new_imgs)-num_neighbours-1):
                for j in range(num_neighbours):
                    new.append([new_imgs[i], new_imgs[i+j+1]])

        # get paths to each pairs of images, and create GT 3D rigid transform between images using first image as reference
        # get GT from each pairs of images and store in array, along with paths to each pair of image
        imgs_list = []
        gts_list = []
        for n1 in new:
            # for both
            imgs = []
            count = 0
            for n2 in n1:
                tmp = n2.split(".")[0]
                data_object = MetaImage(tmp + ".mhd")
                info = data_object.attributes

                # Need to convert to rotation matrix, only need to fix one vector ("vertical axis flip")
                mat = np.reshape(np.array(info["TransformMatrix"].split(" ")).astype(np.float32), (3, 3)) # <- row-wise
                mat = np.transpose(mat) # <- need to read column-wise (!)
                tmp = np.eye(3)
                tmp[1, 1] = -1
                mat = np.matmul(mat, tmp)  # ORDER MATTERS (!) switch and check middle row/column

                # extract offset
                offset = np.array(info["Offset"].split(" ")).astype(np.float32)

                # get transformation matrix (for 3D rigid transformation)
                trans_mat = np.zeros((4, 4))
                trans_mat[:3, :3] = mat
                trans_mat[-1, -1] = 1
                trans_mat[:3, -1] = offset

                if count == 0:
                    inv_trans_mat_first = np.linalg.inv(np.matrix(trans_mat))
                elif count == 1:
                    trans_mat = np.matmul(inv_trans_mat_first, trans_mat) # <- order matters (!)
                    offset = np.array(trans_mat[:3, -1]).flatten()
                    euler_angles = np.array(mat2euler(trans_mat[:3, :3]))  #, cy_thresh=1e-16))
                    # x, y, z = tuple(euler_angles)
                    gt = np.concatenate([offset, euler_angles])

                #val = entropy2(data, base=None)
                imgs.append(n2)
                count += 1
            imgs_list.append(imgs)
            gts_list.append(gt)

        imgs_array = np.array(imgs_list)
        gts_array = np.array(gts_list)

        # filter pairs of images, if one image isn't touching the body
        for i in range(imgs_array.shape[0]):
            if not img_flag:
                # if want to only saved paths
                f = h5py.File(end_path2 + str(i) + '.h5', 'w')
                f.create_dataset("input", data=np.array(imgs_array[i]).astype('S200'), compression="gzip", compression_opts=4)
                f.create_dataset("output", data=gts_array[i], compression="gzip", compression_opts=4)
                f.close()
            else:
                imgs = imgs_array[i]
                gt = gts_array[i]
                # if want to save pairs of images
                input_shape = (1, 256, 256, 2)
                input_im = np.zeros(input_shape[1:], dtype=np.float32)
                for j, img in enumerate(imgs):
                    data_object = MetaImage(img.split(".")[0] + ".mhd")
                    data_tmp = np.asarray(data_object.get_image()).astype(np.float32)
                    input_im[..., j] = resizer(data_tmp, out_dim=(256, 256), gt=False)
                #input_im = input_im.astype(np.uint8)

                f = h5py.File(end_path2 + str(i) + '.h5', 'w')
                f.create_dataset("input", data=input_im, compression="gzip", compression_opts=4)
                f.create_dataset("output", data=gt, compression="gzip", compression_opts=4)
                f.close()



if __name__ == "__main__":

    # folder to store generated data
    num_neighbours = 1 #"3_only"
    img_flag = True
    mode = "10_only" # normal, 3_only, 5_only
    name = "anesthesia_tracking_23_08_num_neighbours_" + str(num_neighbours) + "_images_" + str(img_flag) + "_mode_" + mode

    # paths
    data_path = "/mnt/EncryptedData1/anesthesia/axillary/ultrasound_tracking/"
    end_path = "/home/andrep/workspace/freehand_tracking/data/" + name + "/"

    if not os.path.exists(end_path):
        os.mkdir(end_path)

    # append all patient scan data in a path hierarchy
    tmp = np.sort(os.listdir(data_path))
    locs = []
    for t in tmp:
        if t not in ["2_left_old", "GAN", "1_left", "1_right"]:
            locs.append(t)
    print(locs)

    # run processes in parallel
    proc_num = 16
    p = mp.Pool(proc_num)
    num_tasks = len(locs)
    r = list(tqdm(p.imap(func, locs), "WSI", total=num_tasks))  # list(tqdm(p.imap(func,gts),total=num_tasks))
    p.close()
    p.join()









