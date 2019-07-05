"""Contains dataset importers for NYU Depth Dataset V2 and SYNTHIA-SF"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd
import tables
from skimage import img_as_float32
# from skimage import img_as_float64
from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_ubyte
# from skimage import img_as_uint


RGB = 0
SEGMENTATION = 1
DEPTH = 3

TRAIN = 0
VALIDATION = 1
TEST = 2



class DatasetGenerator:
    """Abstract iterator for looping over elements of a dataset .

    Arguments:
        ratio: ratio of the train-set size to the validation-set size and test-set size
            The first number is for the train-set, the second is for validation-set and what
            is remained is for test-set.(the sum of two numbers should equal to one or less)
        batch_size: Integer batch size.
        repeater: If true, the dataset generator starts generating samples from the beginning when
            it reaches the end of the dataset.
        shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of the
            training.
        output_shape: size of generated images and labels.
        data_type: data type of features.
        label_type: Types of labels to be returned.
    """

    def __init__(self,
                 usage='train',
                 ratio=(1, 0),
                 batch_size=1,
                 repeater=False,
                 shuffle=True,
                 output_shape=None,
                 data_type='float64',
                 label_type=('segmentation', 'instance', 'depth'),
                 **kwargs):
        self.ratio = kwargs[
            'ratio'] if 'ratio' in kwargs else ratio
        self.batch_size = kwargs[
            'batch_size'] if 'batch_size' in kwargs else batch_size
        self.repeater = kwargs['repeater'] if 'repeater' in kwargs else repeater
        self.shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else shuffle
        self.output_shape = kwargs[
            'output_shape'] if 'output_shape' in kwargs else output_shape
        self.data_type = kwargs[
            'data_type'] if 'data_type' in kwargs else data_type
        self.label_type = kwargs[
            'label_type'] if 'label_type' in kwargs else label_type
        self.dataset = self.data_frame_creator()
        self.size = self.dataset.shape[0] - 1
        self.start_index = 0
        self.end_index = np.int32(np.floor(self.ratio[TRAIN] * self.size))
        self.dataset_usage(usage)
        self.index = self.start_index

    def data_frame_creator(self):
        """Pandas dataFrame for addresses of images and corresponding labels"""

        return pd.DataFrame()

    def dataset_usage(self, usage):
        """ Determines the current usage of the dataset:
            - 'train'
            - 'validation'
            - 'test'
        """
        if usage is 'train':
            self.start_index = 0
            self.end_index = np.int32(np.floor(self.ratio[TRAIN] * self.size))
        elif usage is 'validation':
            self.start_index = np.int32(np.floor(self.ratio[TRAIN] * self.size))
            self.end_index = np.int32(np.floor((self.ratio[TRAIN] + self.ratio[VALIDATION])* self.size))
        elif usage is 'test':
            self.start_index = np.int32(np.floor((self.ratio[TRAIN] + self.ratio[VALIDATION])* self.size))
            self.end_index = self.size
        else:
            print('Invalid input for usage variable')
            raise NameError('InvalidInput')

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """Retrieve the next pairs from the dataset"""

        if self.index + self.batch_size > self.end_index:
            if not self.repeater:
                raise StopIteration
            else:
                self.index = self.start_index
        self.index = self.index + self.batch_size

        # loading features(images)
        features = imread(self.dataset.RGB[0])[:, :, :3]

        if self.output_shape is None:
            output_shape = features.shape[:2]
        else:
            output_shape = self.output_shape

        # 1) Resize image to match a certain size.
        # 2) Also the input image is converted (from 8-bit integer)
        # to 64-bit floating point(->preserve_range=False).
        # 3) [:, :, :3] -> to remove 4th channel in png

        features = np.array([
            resize(image=imread(self.dataset.RGB[i])[:, :, :3],
                   output_shape=output_shape,
                   mode='constant',
                   preserve_range=False,
                   anti_aliasing=True)
            for i in range(self.index - self.batch_size, self.index)
        ])

        if self.data_type is 'float32':
            features = img_as_float32(features)

        # loading labels(segmentation)
        segmentation = np.array([
                img_as_ubyte(   
                    resize(image=imread(
                        self.dataset.SEGMENTATION[i]),
                           output_shape=self.output_shape +(1,))/255)
            for i in range(self.index - self.batch_size, self.index)
        ])

        return features, segmentation


class CrossRoad(DatasetGenerator):
    """Iterator for looping over elements of SYNTHIA-SF backwards."""

    def __init__(self, cross_road_dir, **kwargs):

        self.dataset_dir = cross_road_dir
        super().__init__(**kwargs)

    def data_frame_creator(self):
        """ pandas dataFrame for addresses of rgb, depth and segmentation"""

        rgb_dir = self.dataset_dir + "/color/"
        rgb_data = [
            rgb_dir + rgb for rgb in os.listdir(rgb_dir)
        ]

        segmentation_dir =  self.dataset_dir + "/mask/"
        segmentation_data = [
            segmentation_dir + segmentation
            for segmentation in os.listdir(segmentation_dir)
        ]

        dataset = {
            'RGB': rgb_data,
            'SEGMENTATION': segmentation_data
        }

        if self.shuffle:
            return pd.DataFrame(dataset).sample(frac=1, random_state=123)

        return pd.DataFrame(dataset)
