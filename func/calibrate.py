# -*- coding: utf-8 -*-
from email.mime import image
from math import sqrt

__author__ = 'tomek'
#/usr/bin/env python

import cv2
import browse
import sys
import numpy as np
import func.markElements as mark
import operator
from numpy import linalg as LA

def sortByColumn(bigList, *args):
    bigList.sort(key=operator.itemgetter(*args))


def matchMarkers(markerD,markerU,gray,displacement):
    '''
    markerD - punkty wierzchołków na markrze dolnym we współrzednych markera
    markerU - punkty wierzchołków na markrze górnym

    displacement - przesunięcie globalne dolnego obrazu
    '''

    x,y,w,h = markerD[0]
    y += displacement
    canvasMD = gray[y:y+h,x:x+w]

    x,y,w,h = markerU[0]
    canvasMUraw = gray[y:y+h,x:x+w]

    canvasMU = np.zeros_like(canvasMUraw)

    #  lustrazne odbicie górnego obrazu
    for i in range(0,len(canvasMUraw)):
        canvasMU[i] = canvasMUraw[-i]

    heightU = len(canvasMU)
    # przepisanie punktów, dla górnego wraz z transformacją odbicia
    pointsU = [ (p[0],heightU - p[1]) for p in markerU[1].tolist()]
    pointsD = [ (p[0],p[1]) for p in markerD[1].tolist()]

    # pointsD = [(int(p[0]),int(p[1])) for p in pointsD]
    # pointsU = [(int(p[0]),int(p[1])) for p in pointsU]

    # sortowanie punktów wg osi y potem y
    sortByColumn(pointsU,1,0)
    sortByColumn(pointsD,1,0)

    i = 4

    # v1 = calcVector(pointsU,0,1)
    # v2 = calcVector(pointsU,2,3)

    # grupownie punktów w wiersze
    ansU = group(pointsU)
    ansD = group(pointsD)

    p1 = ansU[3][3]
    p2 = ansD[3][3]

    imageDP = []
    imageUP = []
    p1s = []
    p2s = []

    #wybranie wewnętrznych punktów
    for i in range(1,len(ansU[0])-1):
        for j in range(1,len(ansU)-1):
            imageUP.append(ansU[i][j])
            imageDP.append(ansD[i][j])

    # wszystkie punkty
    for i in range(0,len(ansU[0])):
        for j in range(0,len(ansU)):
            p2s.append(ansU[i][j])
            p1s.append(ansD[i][j])

    imageUP = [(p[0],heightU-p[1]) for p in imageUP]

    for mp in imageUP:
        # mark.point(canvasMUraw,(int(mp[0]),int(mp[1])))
        pass

    for mp in imageDP:
        # mark.point(canvasMD,(int(mp[0]),int(mp[1])))
        pass

    # image_points = [np.asarray(imageUP,dtype=np.float32),np.asarray(imageDP,dtype=np.float32)]

    # browse.browse(canvasMUraw)
    f = 'img/results/matching/%d/folder_%d_match_%d_%s.png' % (7,2,33,'down')
    cv2.imwrite(f,canvasMD,[cv2.IMWRITE_PNG_COMPRESSION,0] )
    f = 'img/results/matching/%d/folder_%d_match_%d_%s.png' % (7,2,33,'up')
    cv2.imwrite(f,canvasMUraw,[cv2.IMWRITE_PNG_COMPRESSION,0] )

    # punkty połówek
    global_marker_up_points = [(p[0]+markerU[0][0],p[1]+markerU[0][1]) for p in imageUP]
    global_marker_down_points = [(p[0]+markerD[0][0],p[1]+markerD[0][1]) for p in imageDP]

    image_points = [np.asarray(global_marker_up_points,dtype=np.float32),np.asarray(global_marker_down_points,dtype=np.float32)]

    # p1s-> dół
    # p2s -> gora
    return image_points,canvasMUraw,canvasMD



def calibrate(markerD,markerU,gray,displacement,img,half_size):

    maker_w = marker_h = 5

    (h,w) = half_size
    half_size = (w,h)

    a = np.asarray([(10,10,0),(20,10,0),(10,20,0),(20,20,0)],dtype=np.float32)
    a = np.asarray([(0,0,0),(10,0,0),(20,0,0),(30,0,0),
                    (0,10,0),(10,10,0),(20,10,0),(30,10,0),
                    (0,20,0),(10,20,0),(20,20,0),(30,20,0),
                    (0,30,0),(10,30,0),(20,30,0),(30,30,0)
                   ],dtype=np.float32)


    objectPoints = [a,a]

    #znajdź pary punktów [up, down], down, up
    imagePoints,canvasUp,canvasDown =  matchMarkers(markerD,markerU,gray,displacement)


    canvasMD = gray[displacement:,:]
    mark.points(canvasMD,imagePoints[1])
    # browse.browse(canvasMD)

    # points1 = np.asarray(points1,dtype=np.float32)
    # points2 = np.asarray(points2,dtype=np.float32)

    # [dół,góra] zwiększenie ilośc ipunktóe
    # imagePoints = [points1,points2]

    dist_coefs_0 = np.asarray([(0,0,0,0,0)],dtype=np.float32)

    rms, camera_matrix, dist_coefs, Rotation, Translation = cv2.calibrateCamera(
                                                                        objectPoints,
                                                                        imagePoints,
                                                                        half_size,
                                                                        cameraMatrix=None,
                                                                        distCoeffs=dist_coefs_0,
                                                                        rvecs=None,
                                                                        tvecs=None,
                                                                        flags=(cv2.CALIB_FIX_K3)
                                                                        )

    # R = Rt(Rl)^T
    # T = Tr - R Tl
    Rd = cv2.Rodrigues(Rotation[0])[0]
    Ru = cv2.Rodrigues(Rotation[1])[0]

    cam_pos_d = np.matrix(Rd).T * np.matrix(Translation[0])
    cam_pos_u = np.matrix(Ru).T * np.matrix(Translation[1])

    R = Rd*Ru.T

    T = Translation[0]-cv2.Rodrigues(R)[0]*Translation[1]
    # T = np.asarray((T[0][0],T[1][0],T[2][0]))


    # g2 = cv2.undistort(gray,camera_matrix,dist_coefs)

    img1,img2 = rectify(camera_matrix,dist_coefs,half_size,R,T, img,displacement)


    # R0,jacob0 = cv2.Rodrigues(Rotation[0])
    # R1,jacob1 = cv2.Rodrigues(Rotation[1])

    #macierz fundamntalna (1)
    # R = np.dot(R0,R1.T)
    # Tx,Ty,Tz = Translation[0]
    # A = np.array([[0, -Tz, Ty],[Tz, 0 , -Tx],[-Ty, Tx, 0]])
    # E = np.dot(R,A)
    # inv = LA.inv(camera_matrix)
    # F = np.dot(np.dot(inv.T,E),inv)
    #
    # #macierz fundamntalna (1)
    # F2,q = cv2.findFundamentalMat(points1,points2,method=cv2.FM_8POINT)
    #
    # #macierz fundamntalna (3)
    # rr,cam,dis,cam2,dis2,RRR,TTT,EEE,FFF = cv2.stereoCalibrate(objectPoints,imagePoints,imagePoints,(w,h))
    #
    #
    #
    f = 'img/results/matching/%d/folder_%d_1_%d_%s.png' % (7,17,33,'down')
    cv2.imwrite(f,img1,[cv2.IMWRITE_PNG_COMPRESSION,0] )
    f = 'img/results/matching/%d/folder_%d_2_%d_%s.png' % (7,17,33,'up')
    cv2.imwrite(f,img2,[cv2.IMWRITE_PNG_COMPRESSION,0] )


    return camera_matrix, dist_coefs, R, T


def rectify(camera_matrix,dist_coefs,Size,R,T,gray,displacement):

    canvasMD = gray[displacement:,:]
    (w,h,q) = canvasMD.shape

    R1,R2,P1,P2,Q,qw,qw2 = cv2.stereoRectify(camera_matrix,dist_coefs,
                      camera_matrix,dist_coefs,
                      (w,h),
                      R,
                      T)

    mapx1,mapy1 = cv2.initUndistortRectifyMap(camera_matrix,dist_coefs,R1,P1[:,0:3],(w,h),cv2.CV_32FC1)

    img1 = cv2.remap(canvasMD,mapx1,mapy1,cv2.INTER_LINEAR)

    canvasMU = gray[:displacement,:]
    (w,h,q) = canvasMU.shape

    browse.browse(canvasMU)

    mapx2,mapy2 = cv2.initUndistortRectifyMap(camera_matrix,dist_coefs,R2,P2[:,0:3],(w,h),cv2.CV_32FC1)
    img2 = cv2.remap(canvasMU,mapx2,mapy2,cv2.INTER_LINEAR)

    return img1,img2




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