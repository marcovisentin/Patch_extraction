import os
# import shutil
import random
import nibabel as nib
import numpy
import numpy as np
import pandas as pd
# import keras
# from scipy.ndimage import zoom
# from subprocess import Popen
import scipy.signal
from skimage.measure import label
import matplotlib.pyplot as plt
import warnings
import json
# from . import image_transformations as tr
# import image_transformations as tr
import time
from scipy.stats import multivariate_normal
from skimage.segmentation import find_boundaries
from scipy.spatial.distance import cdist




def load_csv(path, fold=0, x_col_ids=0, y_col_ids=1, check_filenames=True, transpose=True, n_xy=None):
    """
    loads csv files that has columns in the following format:
        x0_fold_0, ..., xj_fold_0, y0_fold_0, ..., yk_fold_0, x0_fold_1, ..., xj_fold_1, y0_fold_1, ..., yk_fold_1 ...
        in this example there are j x columns and k y columns
        the number of entries in each column should be equal to n_subjects (blank entries are not counted)

    Parameters
    ----------
    path: path to the csv file
    fold: the fold index to extract
    x_col_ids: int or list of ints. the indices that correspond to the x's to be extracted in the first fold
        (it is assumed the other folds have the same pattern)
        can be an empty list
    y_col_ids: the same as x_col_ids but for y's
    check_filenames: raises a warning of the same file basename is not in x's and y's when compared to first x
    transpose: see returned variables
    n_xy: int. if None, it is assumed that the total number of x's and y's stored in the csv are
        len(x_col_ids) + len(y_col_ids). if this is not the case, then set the actual value here
        this can be useful if not all the columns need to be extracted


    Returns
    --------
    cases: if len(x_col_ids) == 1 or isinstance(x_col_ids,int), then this is a list of length n_subjects
        if len(x_col_ids) > 1  and transposes == True, then this is a list of lists of shape (n_subjects, len(x_col_ids))
        if len(x_col_ids) > 1  and transposes == False, then this is a list of lists of shape (len(x_col_ids),n_subjects)
    cases_y:  the same as cases but for y's

    """

    def check_cases(cases, cases_y):
        """
           Check if cases x and cases y names correspond
        """
        for k in range(0, len(cases)):
            name = os.path.split(cases[k])[-1]
            name = name.split('.')[0]
            name_y = os.path.split(cases_y[k])[-1]
            name_y = name_y.split('.')[0]
            if not name in name_y:
                warnings.warn(
                    'The filename for ' + cases[k] + ' was not found in the corresponding y target ' + cases_y[
                        k] + '. Ignore if this was intentional.')

    def transpose_list(input_list):

        transposed_list = [list(j) for j in zip(*input_list)]

        return transposed_list

    def extract_column(df, col_id):
        """
            Pandas series (column of the dataset) in list
        """
        col = df.iloc[:, col_id].dropna()
        col = col.values.tolist()

        return col

    def convert_ids_to_list(col_ids):
        """
            Convert list indexes in list
        """
        if isinstance(col_ids, int):
            col_ids_list = [col_ids]
        elif isinstance(col_ids, list):
            col_ids_list = col_ids
        else:
            raise Exception('col_ids must be int or list of ints')
        return col_ids_list

    x_col_ids_list = convert_ids_to_list(x_col_ids)
    y_col_ids_list = convert_ids_to_list(y_col_ids)

    if n_xy is None:
        n_xy = len(x_col_ids_list) + len(y_col_ids_list)

    df = pd.read_csv(path, header=None)

    cases = []
    for x in x_col_ids_list:
        col_id = n_xy*fold + x
        cases.append(extract_column(df, col_id))

    cases_y = []
    for y in y_col_ids_list:
        col_id = n_xy*fold + y
        cases_y.append(extract_column(df, col_id))

    for x in cases:
        # print(x)
        for y in cases_y:
            print(len(x), len(y))
            assert len(x) == len(y), 'Number of cases and corresponding Ys are not equal for fold '+str(fold)+' in '+ path

    if check_filenames:
        for x in cases:
            check_cases(cases[0], x)
        for y in cases_y:
            check_cases(cases[0], y)

    if len(cases) == 1:
        cases = cases[0]
    else:
        if transpose:
            cases = transpose_list(cases)

    if len(cases_y) == 1:
        cases_y = cases_y[0]
    else:
        if transpose:
            cases_y = transpose_list(cases_y)

    return cases, cases_y


def reshape_Y_batch_for_multi_class(Y_in, num_classes, class_ids=None):
    '''

    This function flatten makes y_true consistent with y_pred shape outputted by the model.
    Shaped [batch_size,n_voxels, num_classes]. Necessary to calculate loss function and metrics to be calculated.

    Parameters
    ----------
    Y_in: Augmented mask image, shape (batch_size,n_voxels)
    num_classes: number of classes within the image
    class_ids: None or array
        if None, class_ids will be assumed to be np.arange(0,num_classes)
        if array, num_classes will be overwritten to be len(class_ids)

    Returns
    --------
    Y_final: array, shape (batch_size,n_voxels,num_classes)
        binary mask image with new dimension for classes
    '''

    if class_ids is not None:
        num_classes = len(class_ids)
    else:
        class_ids = np.arange(0, num_classes)

    Y_shape = Y_in.shape
    Y_in = Y_in.astype('float32')
    Y_final = np.zeros((Y_shape[0], Y_shape[1], num_classes))
    for i in range(num_classes):
        Y_new = np.zeros((Y_shape)) + Y_in
        Y_new[Y_new == class_ids[i]] = 1000.5
        Y_new[Y_new != 1000.5] = 0
        Y_new[Y_new == 1000.5] = 1
        Y_final[:, :, i] = Y_new[:, :]

    return Y_final.astype('float32')


def reshape_Y_out_from_multi_class(Y_out, class_ids=None):
    '''

    Loads multi-class output tensor back to 3d array

    Parameters
    ----------
    Y_out: predicted Y within multiple classes, shape (batch_size,n_voxels,num_classes)
    class_ids: None or array of length (Y_out.shape[-1])
        if None, class_ids will be assumed to be np.arange(0,num_classes),
            where num_classes = Y_out.shape[-1]

    Returns
    --------
    Y_final: Multi-label mask image, shape (batch_size,n_voxels,1)
    '''

    print('Before rehsape Y out:', Y_out.shape)
    if class_ids is None:
        num_classes = Y_out.shape[-1]
        class_ids = np.arange(0, num_classes)
    else:
        num_classes = len(class_ids)

    id_dict = dict({k: class_ids[k] for k in range(num_classes)})

    Y_shape = Y_out.shape
    Y_final = np.zeros((Y_shape[0], Y_shape[1], 1))
    # Y_final[0, :, 0] += np.argmax(Y_out[0, :, :], axis=1)
    for j in range(Y_out.shape[0]):
        amax = np.argmax(Y_out[j, :, :], axis=1)
        amax = np.vectorize(id_dict.get)(amax)
        Y_final[j, :, 0] += amax

    return Y_final.astype('float32')


def expand (coord):
    new_coord = [(coord[0]-1, coord[1], coord[2]),(coord[0]+1, coord[1], coord[2]),
                 (coord[0], coord[1]-1, coord[2]),(coord[0], coord[1]+1, coord[2]),
                 (coord[0], coord[1], coord[2]-1),(coord[0], coord[1], coord[2]+1)]
    return new_coord

def get_Manhattan_map(img):
    inout = img.astype(int) * 2 - 1
    # Boundaries coord
    bound = find_boundaries(img) * 1
    coords = np.argwhere(bound).tolist()
    Man_map = np.copy(bound)
    print('Shape map:',Man_map.shape)
    dist = 1
    start = time.time()
    while len(coords) != 0:
        dist = dist + 1
        new_coord_list = []
        for coord in coords:
            for new_coord in expand(coord):
                if(new_coord[0] < Man_map.shape[0] and new_coord[1] < Man_map.shape[1] and new_coord[2] < Man_map.shape[2]) and Man_map[new_coord] == 0:
                    Man_map[new_coord] = dist
                    new_coord_list.append(new_coord)
        coords = list(set(new_coord_list))
    end = time.time()
    print('TIME:', end-start)
    Man_map = Man_map * (-inout) # Boundaries zero
    return Man_map


class Patch_extractor_3D():

    def __init__(self, size, mode='random', n=6, stride=[3, 3, 3]):
        """
        Class for extracting patches

        Parameters
        ----------
        size: list, length number of image dimensions. size of patch to extract
            if the patch size for each axis is smaller than the corresponding image axis size...
                patches will not go outside the image border.
            if the patch size for an axis is bigger than the corresponding image axis size...
                the full length of voxels along that axis will be sampled and padded with zeros
        mode: str. 'random' or 'strided'
            if mode is random, patches will be randomly generated from the image.
            if mode is strided, patches will be approximately evenly sampled from the whole image
        n: int. Number of patches to extract. only has an effect if mode is random
        stride: list, length number of image dimensions. length of stride for patch extraction.
            only has an effect if mode is strided
            this is only an approximate length. it will be altered to make for a more even extraction

        Returns
        --------
        None

        """

        self.size = size
        self.n = n
        self.gen = False
        self.mode = mode
        self.stride = stride

    def generate(self, image):
        """
        generate coordinates for patches

        Parameters
        ----------
        image: array. image to extract patches from

        Returns
        --------
        None

        """

        size = self.size
        shape = image.shape

        if self.mode == 'random':
            n = self.n
            coorStart = np.zeros((len(shape), n), dtype=np.int)
            coorEnd = np.zeros((len(shape), n), dtype=np.int)
            for j in range(len(shape)):
                if shape[j] > size[j]:
                    allowed_range = shape[j]-size[j]
                    # print(allowed_range)
                    axis_coor = np.round(np.random.rand(n)*allowed_range)
                    coorStart[j, :] = np.asarray(axis_coor, dtype=np.int)
                    coorEnd[j, :] = coorStart[j, :]+size[j]
                else:
                    # print(shape[j],size[j])
                    coorStart[j, :] = 0
                    coorEnd[j, :] = shape[j]
            # print(coorStart,coorEnd)

        elif self.mode == 'strided':
            stride = self.stride
            all_axis_coor = []
            size2 = [j for j in size]
            for j in range(len(shape)):
                if shape[j] > size[j]:
                    diff = shape[j]-size[j]

                    # calculating new stride length that would sample the image more evenly
                    n_fit = diff/stride[j] + 1
                    n_fit_new = np.ceil(n_fit)
                    stride_new = int(np.round(diff/(n_fit_new-1)))

                    n_fit = int(np.ceil(diff/stride_new + 1))
                    axis_coor = [j*stride_new for j in range(0, n_fit-1)]
                    axis_coor.append(diff)
                    all_axis_coor.append(axis_coor)
                else:
                    all_axis_coor.append([0])
                    size2[j] = shape[j]

            # print(all_axis_coor)
            n = 1
            for j in all_axis_coor:
                n *= len(j)
            self.n = n
            coorStart = np.zeros((len(shape), n), dtype=np.int)
            coorEnd = np.zeros((len(shape), n), dtype=np.int)
            count = 0

            for x in all_axis_coor[0]:
                for y in all_axis_coor[1]:
                    for z in all_axis_coor[2]:
                        coorStart[0, count], coorStart[1, count], coorStart[2, count] = x, y, z
                        coorEnd[0, count], coorEnd[1, count], coorEnd[2, count] = x+size2[0], y+size2[1], z+size2[2]
                        count += 1
            self.all_axis_coor = all_axis_coor

        self.coorStart = coorStart
        self.coorEnd = coorEnd
        self.or_im_shape = shape
        self.gen = True

    def extract(self, image, regen=False):
        """
        extract the actual patches

        Parameters
        ----------
        image: array. image to extract patches from
        regen: after the extract function has been called once (or if the generator function has already been called),
            subsequent calls of the extract function will use the same coordinates for patch extraction.
            set this to true to regenerate the coordinates

        Returns
        --------
        patches: array, shape (nPatches,patchSize1,patchSize2,patchSize3). extracted patches

        """

        if (not self.gen) or regen:
            self.generate(image)

        coorStart = self.coorStart
        coorEnd = self.coorEnd
        patches = np.zeros((self.n, *self.size))
        # print(patches.shape)
        for j in range(self.n):
            size2 = coorEnd[:, j]-coorStart[:, j]
            # print(size2)
            patches[j, :size2[0], :size2[1], :size2[2]] = image[coorStart[0, j]:coorEnd[0, j],
                                                                coorStart[1, j]:coorEnd[1, j],
                                                                coorStart[2, j]:coorEnd[2, j]]

        return patches

    def get_n_sampled_voxels(self):
        """
        generate an image, of the same shape as the original image from which patches were extracted,
        where the value of each voxel indicates the number of times it was sampled during patch extraction

        Parameters
        ----------

        Returns
        --------
        n_sampled: array. an image, of the same shape as the original image from which patches were extracted,
        where the value of each voxel indicates the number of times it was sampled during patch extraction

        """

        if self.gen:
            coorStart = self.coorStart
            coorEnd = self.coorEnd
            n_sampled = np.zeros(self.or_im_shape)
            for j in range(self.n):
                n_sampled[coorStart[0, j]:coorEnd[0, j],
                          coorStart[1, j]:coorEnd[1, j],
                          coorStart[2, j]:coorEnd[2, j]] += 1
            return n_sampled
        else:
            print('self.extract() or self.generate() have not been run')

    def add_patches_back(self, patches, average=True, average_method='gaussian'):
        """
        reconstruct an image, of the same shape as the original image from which patches were extracted,
        using a given array of patches

        Parameters
        ----------
        patches: array, shape (nPatches,patchSize1,patchSize2,patchSize3).
            must be the same shape as the array that would be output by the extract function,
            since the same generated coordinates will be used to reconstruct
        average: patches are arithmetically added onto their locations,
            and the value for each voxel is subsequently divided by the number of times it was sampled.
            set this to true to skip averaging step.

        Returns
        --------
        im: array. same shape as the original image from which pictures were extracted

        """
        if self.gen:
            coorStart = self.coorStart
            coorEnd = self.coorEnd
            if average:
                if average_method == 'standard':
                    im = np.zeros(self.or_im_shape)
                    for j in range(self.n):
                        size2 = coorEnd[:, j] - coorStart[:, j]
                        im[coorStart[0, j]:coorEnd[0, j],
                        coorStart[1, j]:coorEnd[1, j],
                        coorStart[2, j]:coorEnd[2, j]] += patches[j, :size2[0], :size2[1], :size2[2]]
                        n_sampled = self.get_n_sampled_voxels()
                        im = im / n_sampled
                elif average_method == 'gaussian':
                    im = np.zeros((self.n, *self.or_im_shape))
                    gaussian_weights = np.zeros((self.n, *self.or_im_shape))

                    # Build gaussian filter
                    x, y, z = np.meshgrid(np.linspace(0, self.size[0] - 1, self.size[0]),
                                          np.linspace(0, self.size[1] - 1, self.size[1]),
                                          np.linspace(0, self.size[2] - 1, self.size[2]), indexing = 'ij')
                    xyz = np.column_stack([x.flat, y.flat, z.flat])
                    mu = np.floor(np.array(self.size) / 2)
                    sigma = np.array(self.size) / 6
                    covariance = np.diag(sigma ** 2)
                    gaussian_filter = multivariate_normal.pdf(xyz, mean=mu, cov=covariance)
                    gaussian_filter = gaussian_filter.reshape(x.shape) * 1e5

                    for j in range(self.n):
                        size2 = coorEnd[:, j] - coorStart[:, j]
                        im[j, coorStart[0, j]:coorEnd[0, j],
                        coorStart[1, j]:coorEnd[1, j],
                        coorStart[2, j]:coorEnd[2, j]] = patches[j, :size2[0], :size2[1], :size2[2]]
                        gaussian_weights[j, coorStart[0, j]:coorEnd[0, j],
                        coorStart[1, j]:coorEnd[1, j],
                        coorStart[2, j]:coorEnd[2, j]] = gaussian_filter[:size2[0], :size2[1], :size2[2]]
                    gaussian_weights_norm = gaussian_weights / np.sum(gaussian_weights, axis=0)
                    im = np.sum(im * gaussian_weights_norm, axis=0)
                return im
        else:
            print('self.extract() or self.generate() have not been run')

    def transform_coordinates(self, coordinates):
        """
        given voxel coordinates corresponding to the original image,
        transform these coordinates so that they match the extracted patches

        Parameters
        ----------
        coordinates: array, shape (nCoordinates,3)

        Returns
        --------
        transformed_coordinates: array, shape (nPatches,nCoordinates,3)

        """

        if self.gen:
            [nCoordinates, dim] = coordinates.shape
            coorStart = self.coorStart
            coorEnd = self.coorEnd
            transformed_coordinates = np.zeros((self.n, nCoordinates, dim))
            for j in range(self.n):
                size2 = coorEnd[:, j]-coorStart[:, j]
                transformed_coordinates[j] = coordinates - coorStart[:, j].reshape((1, -1))
            return transformed_coordinates
        else:
            print('self.extract() or self.generate() have not been run')
