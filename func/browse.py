#/usr/bin/env python

'''
browse.py
=========

Sample shows how to implement a simple hi resolution image navigation

Usage
-----
browse.py [image filename]

'''

import numpy as np
import cv2
import sys
from numpy.core.fromnumeric import size


def browse(img):
    """

    :param img:
    """
    small = img
    for i in xrange(3):
        small = cv2.pyrDown(small)

    def onmouse(event, x, y, flags, param):
        h, w = img.shape[:2]
        h1, w1 = small.shape[:2]
        x, y = 1.0 * x * h / h1, 1.0 * y * h / h1
        zoom = cv2.getRectSubPix(img, (800, 600), (x + 0.5, y + 0.5))
        cv2.imshow('zoom', zoom)

    cv2.imshow('preview', small)
    cv2.setMouseCallback('preview', onmouse)
    cv2.waitKey()
    cv2.destroyAllWindows()
