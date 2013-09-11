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
            img[points[0][:,0],points[0][:,1]] = (255,255,255)
    return img


def singleContour(img,cnt):
    for c in cnt:
        img[c] = (0,255,0)

    return img


def corners(img,corners):
    '''
    wierzcholki na niebiesko
    '''
    for corn in corners:
        cv2.circle(img, (corn[0],corn[1]),5,(255,0,0,0),2)

    return img


def point(img,point):
    '''
    skrajne wierzchołki na blekitno
    '''
    cv2.circle(img, point ,5,(255,255,0,0),2)

    return img


def YellowPoint(img,point):
    cv2.circle(img, point ,10,(0,255,244,0),5)

    return img


def points(img,points):
    '''
    na biało
    '''
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

def drawHoughLines(lines,img):

    # dla kolorowych obrazow sa 3 wymiary , 3 jest zbędny nam potem
    m,n,w = img.shape
    factor = 0
    for (rho, theta) in lines[:4]:
        # blue for infinite lines (only draw the 5 strongest)
        x0 = np.cos(theta)*rho
        y0 = np.sin(theta)*rho
        pt1 = ( int(x0 + (m+n)*(-np.sin(theta))), int(y0 + (m+n)*np.cos(theta)) )
        pt2 = ( int(x0 - (m+n)*(-np.sin(theta))), int(y0 - (m+n)*np.cos(theta)) )
        cv2.line(img, pt1, pt2, (128,0,128), 2)
        factor += 50
    return img

def drawPoly(img,poly):

    points = poly.points
    pairs = [(points[i],points[i+1]) for i in range(0,len(points)-1)]
    for pt1,pt2 in pairs:
        img = drawSegment(img,pt1,pt2)

    return img

def drawSegment(img,p1,p2):
    cv2.line(img,p1,p2,(255,255,255),4)
    return img