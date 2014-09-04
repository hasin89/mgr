# -*- coding: utf-8 -*-
from math import sqrt

__author__ = 'tomek'
# /usr/bin/env python

import cv2
import browse
import sys
import numpy as np


def Point(img, point, size=5, color=(255, 255, 0, 0), thickness=2):
    '''
    skrajne wierzchołki na blekitno
    '''
    cv2.circle(img, point , size , color, thickness)

    return img

def Poly(img, poly):
    """
    :param img: Mat image
    :param poly: list of points
    """
    points = poly.points
    pairs = [(points[i], points[i + 1]) for i in range(0, len(points) - 1)]
    for pt1, pt2 in pairs:
        img = Segment(img, pt1, pt2)

    return img

def Segment(img, p1, p2, color=(255, 255, 255), thickness=4):
    """
    :param img: Mat image
    :param p1: tuple (x1,y1)
    :param p2: tuple (x2,y2)
    :param color: tuple (red,green,blue)
    :param thickness: int
    """
    cv2.line(img, p1, p2, color, thickness)
    return img
