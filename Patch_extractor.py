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
