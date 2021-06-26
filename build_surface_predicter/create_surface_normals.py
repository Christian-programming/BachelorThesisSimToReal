# Copyright 2021
# Author: Christian Leininger <info2016frei@gmail.com>

import cv2
import numpy as np


def create_normal(d_im, distance=1):
    """  Computes from a depth image the corresponding surface
    Args:
            param1(numpy array): d_im depth image size 256x256
            param2(int): distance value of the distance of the two points
    Returns: numpy array of size 256x256x3 of surface normals
    """
    d_im = d_im.astype("float64")
    data_type = "float64"
    normals = np.zeros((256, 256, 3), dtype=data_type)
    h, w = d_im.shape
    for i in range(distance, w-distance):
        for j in range(distance, h-distance):
            t = (d_im[j, i + distance] - d_im[j, i - distance]) / (2.0 * distance)
            f = (d_im[j + distance, i] - d_im[j - distance, i]) / (2.0 * distance)
            direction = np.array([-t, -f, 1])
            magnitude = np.sqrt(t**2 + f**2 + 1)
            n = direction / magnitude
            normals[j, i, :] = n
    return normals
