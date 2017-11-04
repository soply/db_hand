# coding: utf8

""" Python file with methods to handle the Cropped Yale Face Database B.

Remark
----------
Current data contains 39 different subjects, each under 64 illumination
conditions. However, the different poses are missing currently.

References
-----------
[1] Lee, Kuang-Chih, Jeffrey Ho, and David J. Kriegman. "Acquiring linear subspaces
    for face recognition under variable lighting." IEEE Transactions on pattern
    analysis and machine intelligence 27.5 (2005): 684-698.


"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float, img_as_int
from skimage.transform import rescale

# Get basepath such that only relatives paths matter from this folder on
basepath = os.path.dirname(os.path.realpath(__file__))

def read_single(nr_subject, first_angle, second_angle, scale=1.0,
                datatype="float"):
    """ Read and return a single face with the given subject nr, and first and
    second angle's of illumination. The inputs are preprocessed before reading
    the face, meaning that integer values will be transformed into the correct
    strings.

    Parameters
    ------------
    nr_subject : Integer
        Number of the subject that should be loaded, e.g. nr_subject = 1.

    first_angle : Integer
        Signed first illumination angle, e.g. first_angle = -5.

    second_angle : Integer
        Signed second illumination angle, e.g. second_angle = -40.

    scale : Float in (0, 1)
        Float value passed to skimage.scale to rescale the images to a smaller
        size if desired.

    datatype : string
        Datatype of the image. If datatype is "int", the images will remain
        unchanged. If datatype is "float" the images will converted to floating
        point values in [0,1].

    Returns
    ------------
    Numpy array with image values of the face. Datatype depends on input
    argument (see datatype).

    Example
    -----------
    I = read_single(1, -70, 0, datatype = "float")
    plt.imshow(I, cmap='gray')
    plt.show()

    Reads face from file CroppedYaleFaces/yaleB01/yaleB01_P00A-070E+00.pgm as
    numpy array (float in [0,1]).
    """
    nr_subject = str(nr_subject).zfill(2)
    if first_angle < 0:
        first_angle = "-{0}".format(str(first_angle)[1:].zfill(3))
    else:
        first_angle = "+{0}".format(str(first_angle).zfill(3))
    if second_angle < 0:
        second_angle = "-{0}".format(str(second_angle)[1:].zfill(2))
    else:
        second_angle = "+{0}".format(str(second_angle).zfill(2))
    loadstr = basepath+"/CroppedYaleFaces/yaleB{0}/yaleB{0}_P00A{1}E{2}.pgm".format(
        nr_subject, first_angle, second_angle)
    return read_single_filename(loadstr, scale, datatype)


def read_single_filename(fn, scale=1.0, datatype="float"):
    """ Read and return a single face from given filename. Returns it as a
    numpy array.

    Parameters
    ------------
    fn : string
        Relative filename from folder containing the folder 'CroppedYaleFaces'.

    scale : Float in (0, 1)
        Float value passed to skimage.scale to rescale the images to a smaller
        size if desired.

    datatype : string
        Datatype of the image. If datatype is "int", the images will remain
        unchanged. If datatype is "float" the images will converted to floating
        point values in [0,1].

    Returns
    ------------
    Numpy array with image values of the face. Datatype depends on input
    argument (see datatype).

    Example
    -----------
    I = read_single_filename('CroppedYaleFaces/yaleB01/yaleB01_P00A-070E+00.pgm',
                             datatype = "float")
    plt.imshow(I, cmap='gray')
    plt.show()

    Reads face from file CroppedYaleFaces/yaleB01/yaleB01_P00A-070E+00.pgm as
    numpy array (float in [0, 1]).
    """
    with open(fn) as f:
        data = np.array(read_pgm(f)).astype('int16')
    if datatype == "float":
        data = img_as_float(img_as_int(data))
    return rescale(data, scale)


def read_ambient(nr_subject, scale=1.0, datatype="float"):
    """ Read and return a single ambient image related to a specific subject.
    The input is preprocessed before reading the face, meaning that integer
    values will be transformed into the correct strings.

    Parameters
    ------------
    nr_subject : Integer
        Number of the subject that should be loaded, e.g. nr_subject = 1.

    scale : Float in (0, 1)
        Float value passed to skimage.scale to rescale the images to a smaller
        size if desired.

    datatype : string
        Datatype of the image. If datatype is "int", the images will remain
        unchanged. If datatype is "float" the images will converted to floating
        point values in [0,1].

    Returns
    ------------
    Numpy array with image values of the ambient image. Datatype depends on
    input argument (see datatype).

    Example
    -----------
    I = read_ambient(1)
    plt.imshow(I, cmap='gray')
    plt.show()

    Reads ambient image from file CroppedYaleFaces/yaleB01/yaleB01_P00A_Ambient.pgm
    as numpy array.
    """
    nr_subject = str(nr_subject).zfill(2)
    loadstr = basepath+"/CroppedYaleFaces/yaleB{0}/yaleB{0}_P00_Ambient.pgm".format(
        nr_subject)
    with open(loadstr) as f:
        data = np.array(read_pgm(f)).astype('int16')
    if datatype == "float":
        data = img_as_float(img_as_int(data))
    return rescale(data, scale)


def read_subject_all(nr_subject, scale=1.0, datashape="columns",
                     datatype="float"):
    """ Reads and returns all images for a single subject with the given number.

    Parameters
    --------------
    nr_subject : Integer
        Number of the subject that should be loaded, e.g. nr_subject = 1.

    scale : Float in (0, 1)
        Float value passed to skimage.scale to rescale the images to a smaller
        size if desired.

    datashape : string
        Defines how to data is returned. If datashape = "columns" than each
        single image resembles a column and we return a matrix of size
        (n_pixel, n_images). If datashape = "matrices", we do not modify the
        image but return a 3D array with dimensions
        (n_images, n_pixel_x, n_pixel y).

    datatype : string
        Datatype of the image. If datatype is "int", the images will remain
        unchanged. If datatype is "float" the images will converted to floating
        point values in [0,1].

    Returns
    -------------
    Returns the data object containing all images of the respective subject.
    Note that the ambient image of the database is left out. Moreover, the
    representation/shape of the returned object depends on the chosen datashape.

    Example
    -------------
    In [3]: data = read_subject_all(1, datashape = "columns", datatype = "float")
            # data[1] contains (n_pixel, n_images) matrix with all images related
            # to subject 1
    """
    nr_subject = str(nr_subject).zfill(2)
    files = glob.glob(basepath+'/CroppedYaleFaces/yaleB{0}/*.pgm'.format(nr_subject))
    # Remove ambient image
    files = [item for item in files if "Ambient" not in item]
    # Get first image for extracting data format
    I = read_single_filename(files[0], scale, datatype)
    if datashape == "matrices":
        data = np.zeros((len(files), I.shape[0], I.shape[1]))
        for i, fn in enumerate(files):
            data[i, :, :] = read_single_filename(fn, scale, datatype)
    elif datashape == "columns":
        data = np.zeros((I.shape[0] * I.shape[1], len(files)))
        for i, fn in enumerate(files):
            data[:, i] = read_single_filename(fn, scale, datatype).ravel()
    return data


def read_all(scale=1.0, datashape="columns", datatype="float",
             until_subject = 100):
    """ Reads and returns all images of the database. Format of returned
    python dict depends on the input datashape.

    Parameters
    --------------
    scale : Float in (0, 1)
        Float value passed to skimage.scale to rescale the images to a smaller
        size if desired.

    datashape : string
        Defines how to data is returned. If datashape = "columns" than each
        single image resembles a column and we return a python dict where each
        of the n_subjects entries contains a numpy array of the shape
        (n_pixel, n_images).
        If datashape = "matrices", we do not modify the images and return a
        python dict where each of the n_subjects keys contains a
        (n_images, n_pixel_x, n_pixel y) numpy array.

    datatype : string
        Datatype of the image. If datatype is "int", the images will remain
        unchanged. If datatype is "float" the images will converted to floating
        point values in [0,1].

    until_subject : Integer
        Can be used as an upper boundary for subjects considered. Note that this
        does not have an effect if it exceeds the number of faces in the
        data base.

    Returns
    -------------
    Returns the data object containing all images of the database.
    Note that the ambient image of each subject of the database is left out.
    Moreover, the representation/shape of the returned object depends on the
    chosen datashape.

    Example
    -------------
    In [3]: data = read_all(datashape = "columns", datatype = "float")
    In [4]: data.shape
    Out[4]: (39, 32256, 64)
    """
    retr = {}
    counter = 1
    subject_dirs = [item for item in os.listdir(basepath+"/CroppedYaleFaces/")
                        if item[0:4] == 'yale']
    print "Loading database..."
    while len(subject_dirs) > 0 and counter <= until_subject:
        if os.path.isdir(basepath+"/CroppedYaleFaces/yaleB{0}".format(
                str(counter).zfill(2))):
            retr[counter] = read_subject_all(counter, scale, datashape,
                                             datatype)
            subject_dirs.remove("yaleB{0}".format(str(counter).zfill(2)))
            print "Loading data ", counter
        counter += 1
    return retr

def get_image_format_for_scale(scale=1.0):
    """ Function to get the image shape. Uses the image
    CroppedYaleFaces/yaleB01/yaleB01_P00A-070E+00.pgm, loads this for the given
    scale and returns its dimensionalities.

    Parameters
    -------------
    scale : Float in (0, 1)
        Float value passed to skimage.scale to rescale the images to a smaller
        size if desired.

    Returns
    ------------
    Shape of the image at the given scale.
    """
    img = read_single(1, -70, 0, scale)
    return img.shape


def read_pgm(pgmf):
    """ Return a raster of integers from a PGM as a list of lists.

    Parameters
    ------------
    pgmf: Python file object
        File object that opened a pgm file.

    Returns
    -------------
    List of lists with integers containing the content of the pgm file.

    Source
    -----------
    http://stackoverflow.com/questions/35723865/read-a-pgm-file-in-python
    """
    assert pgmf.readline() == 'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255
    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster
