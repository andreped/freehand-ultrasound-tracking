import vtk
import os
from keras_tools.metaimage import *
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
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


# folder to store generated data
num_neighbours = 1
img_flag = True
name = "anesthesia_tracking_23_08_num_neighbours_" + str(num_neighbours) + "_images_" + str(img_flag)

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


colors = vtk.vtkNamedColors()

for new in new_imgs[100:]:
    input_shape = (256, 256)
    data_object = MetaImage(new.split(".")[0] + ".mhd")
    data_tmp = np.asarray(data_object.get_image()).astype(np.float32)
    data_matrix = resizer(data_tmp, out_dim=(256, 256), gt=False).astype(np.uint8)

    data_matrix = np.flip(data_matrix, axis=0)

    plt.imshow(data_matrix, cmap="gray")
    plt.show()

    out_size = (256, 256)
    #data_matrix = cv2.resize(data_matrix, out_size)

    # For VTK to be able to use the data, it must be stored as a VTK-image.
    #  This can be done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    # The previously created array is converted to a string of chars and imported.
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    # The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    # Because the data that is imported only contains an intensity value
    #  (it isnt RGB-coded or someting similar), the importer must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
    # The following two functions describe how the data is stored and the dimensions of the array it is stored in.
    #  For this simple case, all axes are of length 75 and begins with the first element.
    #  For other data, this is probably not the case.
    # I have to admit however, that I honestly dont know the difference between SetDataExtent()
    #  and SetWholeExtent() although VTK complains if not both are used.
    dataImporter.SetDataExtent(0, out_size[0]-1, 0, out_size[1]-1, 0, 1)  #0, 255, 0, 255, 0, 255)
    dataImporter.SetWholeExtent(0, out_size[0]-1, 0, out_size[1]-1, 0, 1)  #0, 255, 0, 255, 0, 255)

    # The following class is used to store transparency-values for later retrival.
    #  In our case, we want the value 0 to be
    # completely opaque whereas the three different cubes are given different transparency-values to show how it works.
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    #alphaChannelFunc.AddPoint(0, 0.0)
    #alphaChannelFunc.AddPoint(50, 0.05)
    #alphaChannelFunc.AddPoint(100, 0.1)
    #alphaChannelFunc.AddPoint(150, 0.2)
    #alphaChannelFunc.AddPoint(255, 0.5)


    # This class stores color data and can create color tables from a few color points.
    #  For this demo, we want the three cubes to be of the colors red green and blue.
    colorFunc = vtk.vtkColorTransferFunction()
   # colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
    #colorFunc.AddRGBPoint(50, 1.0, 0.0, 0.0)
    #colorFunc.AddRGBPoint(100, 0.0, 1.0, 0.0)
    #colorFunc.AddRGBPoint(150, 0.0, 0.0, 1.0)

    # The previous two classes stored properties.
    #  Because we want to apply these properties to the volume we want to render,
    # we have to store them in a class that stores volume properties.
    volumeProperty = vtk.vtkVolumeProperty()
    #volumeProperty.SetColor(colorFunc)
    #volumeProperty.SetScalarOpacity(alphaChannelFunc)

    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()#vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    # The class vtkVolume is used to pair the previously declared volume as well as the properties
    #  to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # With almost everything else ready, its time to initialize the renderer and window, as well as
    #  creating a method for exiting the application
    #renderer = vtk.vtkRenderer()
    renderer = vtk.vtkOpenGLRenderer()
    window = vtk.vtkRenderWindow()
    window.SetSize(1000, 1000)
    window.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(window)

    # We add the volume to the renderer ...
    renderer.AddVolume(volume)
    renderer.SetBackground(colors.GetColor3d("White"))

    # ... and set window size.
    #window.SetSize(400, 400)


    # A simple function to be called when the user decides to quit the application.
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)


    # Tell the application to use the function as an exit check.
    window.AddObserver("AbortCheckEvent", exitCheck)

    renderInteractor.Initialize()
    # Because nothing will be rendered without any input, we order the first render manually
    #  before control is handed over to the main-loop.
    window.Render()
    renderInteractor.Start()
