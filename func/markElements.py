# -*- coding: utf-8 -*-
from math import sqrt

__author__ = 'tomek'
#/usr/bin/env python

import cv2
import browse
import sys
import numpy as np

def contoursMap(img_shape,contours):

    #kolorowanie kontrów
    nimg = np.zeros((img_shape[0],img_shape[1],3),dtype='uint8')
    nimg[:][:][:] = 0

    for c in contours[0]:
        nimg[c] = (0,255,0)

    return nimg

def contours(img,contours):
    for c in contours.itervalues():
        if len(c)>0:
            points = np.asarray([c])
            img[points[0][:,0],points[0][:,1]] = (0,255,0)
    return img


def singleContour(img,cnt):
    for c in cnt:
        img[c] = (0,255,0)

    return img


def corners(img,corners):
    for corn in corners:
        cv2.circle(img, (corn[0],corn[1]),5,(255,0,0,0),2)

    return img


def point(img,point):
    cv2.circle(img, point ,5,(255,255,0,0),2)

    return img


def YellowPoint(img,point):
    cv2.circle(img, point ,10,(0,255,244,0),5)

    return img


def points(img,points):
    for point in points.itervalues():
        cv2.circle(img, point ,4,(255,255,255,0),3)

    return img


def object(img,mainCNT):

    r = mainCNT
    A = (r[0],r[1])
    B = (r[0]+r[2],r[1]+r[3])
    cv2.rectangle(img,A,B,(255,0,0),2)
    # cv2.drawContours(img,mainCNT,-1,(255,0,0),2)

    return img