# -*- coding: utf-8 -*-
__author__ = 'tomek'
#/usr/bin/env python

'''

gamma correction

'''

import cv2
import numpy as np
import sys


def correction(gray, gammaFactor):
    u"""
    skala szarości
    :param gray:
    :param gammaFactor:
    :return:
    """
    gamma_correction = 1.0 / gammaFactor

    img_tmp = gray / 255.0
    cv2.pow(img_tmp, gamma_correction, img_tmp)
    img_gamma = img_tmp * 255.0

    # zamiana na int
    img_result = np.array(img_gamma, 'uint8')
    return img_result
