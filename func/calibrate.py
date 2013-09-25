# -*- coding: utf-8 -*-
from math import sqrt

__author__ = 'tomek'
#/usr/bin/env python

import cv2
import browse
import sys
import numpy as np
import func.markElements as mark
import operator

def sortByColumn(bigList, *args):
    bigList.sort(key=operator.itemgetter(*args))

def calibrate(markerD,markerU,gray,displacement):
    x,y,w,h = markerD[0]
    y += displacement
    canvasMD = gray[y:y+h,x:x+w]

    x,y,w,h = markerU[0]
    canvasMUraw = gray[y:y+h,x:x+w]

    canvasMU = np.zeros_like(canvasMUraw)

    for i in range(0,len(canvasMUraw)):
        canvasMU[i] = canvasMUraw[-i]

    heightU = len(canvasMU)
    pointsU = [ (p[0],heightU - p[1]) for p in markerU[1].tolist()]
    pointsD = [ (p[0],p[1]) for p in markerD[1].tolist()]

    # pointsD = [(int(p[0]),int(p[1])) for p in pointsD]
    # pointsU = [(int(p[0]),int(p[1])) for p in pointsU]

    sortByColumn(pointsU,1,0)
    sortByColumn(pointsD,1,0)

    i = 4

    v1 = calcVector(pointsU,0,1)
    v2 = calcVector(pointsU,2,3)
    ansU = group(pointsU[2:-2])
    ansD = group(pointsD[2:-2])

    p1 = ansU[3][3]
    p2 = ansD[3][3]

    imageP = []
    mirrorP = []

    #wybranie wewnętrznych punktów
    for i in range(1,len(ansU[0])-1):
        for j in range(1,len(ansU)-1):
            mirrorP.append(ansU[i][j])
            imageP.append(ansD[i][j])

    mirrorP = [(p[0],heightU-p[1]) for p in mirrorP]

    for mp in mirrorP:
        mark.point(canvasMUraw,(int(mp[0]),int(mp[1])))

    for mp in imageP:
        mark.point(canvasMD,(int(mp[0]),int(mp[1])))

    # browse.browse(canvasMUraw)
    f = 'img/results/matching/%d/folder_%d_match_%d_%s.png' % (6,16,33,'down')
    cv2.imwrite(f,canvasMD,[cv2.IMWRITE_PNG_COMPRESSION,0] )
    f = 'img/results/matching/%d/folder_%d_match_%d_%s.png' % (6,16,33,'up')
    cv2.imwrite(f,canvasMUraw,[cv2.IMWRITE_PNG_COMPRESSION,0] )

    pass

def calcVector(list,key1,key2):
    vec = [list[key1][0]-list[key2][0],list[key1][1]-list[key2][1]]
    if vec[1] != 0:
        # vec[0] = vec[0]/vec[1]
        # vec[1] = vec[1]/vec[1]
        pass
    return vec

def group(points,axis=1):
    ans = [points[k+1][axis]-points[k][axis] for k in range(0,len(points)-1)]
    mean = np.mean(ans)*2
    indexes = [i for i,v in enumerate(ans) if v>mean]

    p = []
    p.append( points[:indexes[0]+1] )
    for i in range(0,len(indexes)-1):
        p.append(points[indexes[i]+1:indexes[i+1]+1])
    p.append( points[indexes[-1]+1:] )

    for line in p:
        sortByColumn(line,0)

    return p