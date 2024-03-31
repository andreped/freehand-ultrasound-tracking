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



if __name__ == "__main__":

    # folder to store generated data
    num_neighbours = 1
    img_flag = True
    name = "anesthesia_tracking_23_08_num_neighbours_" + str(num_neighbours) + "_images_" + str(img_flag)

    mode = 1 # 0 or 1, 0 uses fixed image in pairs as reference, 1 uses first image in sequence as reference

    # paths
    data_path = "/mnt/EncryptedData1/anesthesia/axillary/ultrasound_tracking/"
    end_path = "/home/andrep/workspace/freehand_tracking/data/" + name + "/"

    if not os.path.exists(end_path):
        os.mkdir(end_path)

    # append all patient scan data in a path hierarchy
    tmp = np.sort(os.listdir(data_path))
    locs = []
    for t in tmp:
        if t not in ["2_left_old", "GAN"]:
            locs.append(t)
    print(locs)

    for pat in locs:
        print(pat)
        curr1 = data_path + pat
        end_path1 = end_path + pat + "/"
        for scan in os.listdir(curr1):  # type of scan: linear, back&forth linear, natural, crazy
            print(scan)
            if scan not in ["linear", "linear-back-and-forth"]:
                continue
            end_path2 = end_path1 + scan + "/"

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

            # creates non-adjacent pairs (jumps over nearest)
            new = []
            for i in range(len(new_imgs) - num_neighbours - 1):
                for j in range(num_neighbours):
                    new.append([new_imgs[i], new_imgs[i + j + 1]])

            print(new)

            # get paths to each pairs of images, and create GT 3D rigid transform between images using first image as reference
            # get GT from each pairs of images and store in array, along with paths to each pair of image
            imgs_list = []
            gts_list = []
            flag = True
            for n1 in new:
                # for both
                imgs = []
                count = 0
                print("---")
                print(n1)
                for n2 in n1:
                    tmp = n2.split(".")[0]
                    data_object = MetaImage(tmp + ".mhd")
                    info = data_object.attributes

                    # Need to convert to rotation matrix, only need to fix one vector ("vertical axis flip")
                    print(info)
                    print(info["TransformMatrix"])
                    mat = np.reshape(np.array(info["TransformMatrix"].split(" "), dtype=np.float64), (3, 3))
                    print(mat)
                    mat = np.transpose(mat)
                    print(mat)
                    print(np.linalg.det(mat))
                    tmp = np.eye(3)
                    tmp[1, 1] = -1
                    #tmp = np.linalg.inv(tmp)
                    print(tmp)
                    mat = np.matmul(mat, tmp) # ORDER MATTERS (!) switch and check middle row/column
                    #mat = np.matmul(tmp, mat)

                    print(mat)
                    print(np.linalg.det(mat))

                    # extract offset
                    offset = np.array(info["Offset"].split(" ")).astype(np.float32)
                    print(offset)

                    # get transformation matrix (for 3D rigid transformation)
                    trans_mat = np.zeros((4, 4))
                    trans_mat[:3, :3] = mat
                    trans_mat[-1, -1] = 1
                    trans_mat[:3, -1] = offset
                    print()
                    print(trans_mat)

                    if count == 0:
                        if flag:
                            inv_trans_mat_first = np.linalg.inv(trans_mat)
                        if mode == 1:
                            flag = False
                        print()
                    elif count == 1:
                        trans_mat = np.matmul(inv_trans_mat_first, trans_mat)  # <- maybe order matter? Makes sense to put inverse matrix first
                        offset = np.array(trans_mat[:3, -1]).flatten()
                        euler_angles = np.array(mat2euler(trans_mat[:3, :3]))  # , cy_thresh=1e-16)) # <- should I hard-code this parameter or use the default adaptive one?
                        #x, y, z = tuple(euler_angles)
                        gt = np.concatenate([offset, euler_angles])

                    # val = entropy2(data, base=None)
                    imgs.append(n2)
                    count += 1
                imgs_list.append(imgs)
                gts_list.append(gt)

                #exit()



            imgs_array = np.array(imgs_list)
            gts_array = np.array(gts_list)

            print(gts_array)


            # if use fixed adjacent image as reference in each pair
            if mode == 0:
                print(scan)
                x = 0
                z = 0
                x_vals = []
                z_vals = []
                for gt in gts_array:
                    x += gt[0]
                    z += gt[2]
                    x_vals.append(x)
                    z_vals.append(z)
            elif mode == 1:
                # if use first image as reference
                x_vals = gts_array[:, 0]
                z_vals = gts_array[:, 2]


            plt.plot(z_vals, x_vals)
            plt.xlabel("z")
            plt.ylabel("x")
            plt.scatter([z_vals[0]], [x_vals[0]], c="r")
            plt.scatter([z_vals[-1]], [x_vals[-1]], c="orange")
            plt.title("Patient: " + pat + ", scan: " + scan)
            x_range = max(x_vals) - min(x_vals)
            z_range = max(z_vals) - min(z_vals)
            #plt.xlim([min(z_vals), max(z_vals)])
            plt.ylim([min(x_vals) + np.abs(x_range)/2 - np.abs(z_range)/2, max(x_vals) - np.abs(x_range)/2 + np.abs(z_range)/2])
            plt.show()




