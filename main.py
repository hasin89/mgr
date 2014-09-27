# -*- coding: utf-8 -*-
from numpy.lib.function_base import average
from func import histogram
from func.analise import getInnerSegments
from scene import edge

__author__ = 'tomek'
#/usr/bin/env python

'''
This sample demonstrates Canny edge detection.

Usage:
  edge.py [<video source>]

  Trackbars control edge thresholds.

'''

import cv2
import func.browse as browse
import numpy as np
import func.trackFeatures as features
import func.markElements as mark
import func.analise as an
import func.calibrate as ca
import func.histogram


def markCounturResult(img,edge):
    vis_filtred = img.copy()

    vis_filtred[edge != 0] = (0, 255, 0)

    return vis_filtred

# default gauss_kernel = 11
# default gamma factor = 0.45


def markFeatures(src,stuff):

    (myObject,markerSqrBnd) = stuff
    cornerList = myObject.corners
    mainBND = myObject.sqrBnd
    contours = myObject.contours
    longestContour = myObject.longestContour
    left = myObject.left
    rigth = myObject.right
    lines = myObject.lines
    crossing = myObject.crossing
    poly = myObject.poly
    innerSegments = myObject.innerSegments
    innerLines = myObject.innerLines
    
    
    img = src.copy()
    # img[:][:] = (0,0,0)

    # zaznaczenie krawędzi na biało
    img = mark.contours(img,contours)

    #zaznaczenie najdluższego znalezionego konturu na zielono
    # img = mark.singleContour(img,longestContour)

    #zaznaczenie wierzchołków na niebiesko
    img = mark.corners(img,cornerList)

    #zaznaczenie interesujących miejsc (obiektu glownego na niebiesko
    img = mark.object(img,mainBND)
    img = mark.object(img,markerSqrBnd)

    # img = mark.point(img,marker)
    # img = mark.point(img,right)

    #to zaznaczało srodki ciezkosci na biaol
    # img = mark.points(img,centrum)

    #zaznaczanie centrum obrazu na żółto
    xc0 = img.shape[0]/2
    yc0 = img.shape[1]/2
    point = (yc0,xc0)
    # img = mark.YellowPoint(img,point)

    # for p in innerSegments:
        # img = mark.YellowPoint(img,p.points[0])
        # img = mark.YellowPoint(img,p.points[1])
        # img = mark.drawSegment(img,p.points[0],p.points[1])

        # zazmacz lini hougha
    mark.drawHoughLines(lines,img)
    mark.drawHoughLines(innerLines,img)
    # img = mark.drawPoly(img,poly)


    return img


def markExtremePoints(img,edge):
    '''
        zaznacza punkty o max i nim wartościach współrzednych x i y
    :param img: obraz
    :param edge: wynik algorytmy Canny
    '''

    # wyznaczene ekstremalnych punktow

    # points = [(x,y) for (x,l) in enumerate(edge_filtred) for (y,v) in enumerate(l) if v==1]
    # points = [point for point in edge_filtred if point != 0]

    points = np.nonzero(edge)
    maxXIndex = np.where(points[0] == max(points[0]))[0][0]
    maxYIndex = np.where(points[1] == max(points[1]))[0][0]
    minXIndex = np.where(points[0] == min(points[0]))[0][0]
    minYIndex = np.where(points[1] == min(points[1]))[0][0]

    # zanaczenie ekstremalnych punktow kolkami

    cv2.circle(img, (points[1][maxXIndex],points[0][maxXIndex]),20,(0,255,0,0),10)
    cv2.circle(img, (points[1][maxYIndex],points[0][maxYIndex]),20,(0,255,0,0),10)
    cv2.circle(img, (points[1][minXIndex],points[0][minXIndex]),20,(0,255,0,0),10)
    cv2.circle(img, (points[1][minYIndex],points[0][minYIndex]),20,(0,255,0,0),10)

    return img

def countNonZeroRowsY(edge):
    peri= []
    for i in range(edge.shape[0]):
        p = len(np.nonzero(edge[i])[0])
        peri.append(p)
    #pierwsze kontury
    non = np.nonzero(peri)
    a,b = non[0][0],non[0][-1]

    # margin = int(len(peri)*0.1)
    # peri2 = np.asarray(peri[margin:-margin])
    # maxIndex = np.where(peri2 == max(peri2))[0][0] + margin
    # maxValue = max(peri2)

    # mirror_line = [i for i in range(maxIndex-100,maxIndex+101) if peri[i] > maxValue*0.1]

    # smooth = np.convolve(peri,[1,1,1],'same')

    # peri2 = []
    # j=1;
    # for i in range(a,b+1):
    #     p2 = 0
    #     for q in range(-j,j+1):
    #         p2 += peri[i+q]
    #     p2 = p2 * (1.0 / (2*j+1))
    #     peri2.append(int(p2))

    # grad = np.gradient(np.asarray(peri2))
    # grad = np.array(grad, 'uint8')


    h,a,b = histogram.draw2(peri)
    return h,non[0][0],non[0][-1],[a,b]

def countNonZeroRowsX(edge):
    peri= []
    for i in range(edge.shape[1]):
        peri.append(len(np.nonzero(edge[:,i])[0]))
    non = np.nonzero(peri)
    c,d = non[0][0],non[0][-1]
    h = histogram.draw(peri[c:d])
    return h,c,d

def loadImage(filename):
    print(filename)
    imgT = cv2.imread(filename)
    shape = (round(0.25*imgT.shape[1]),round(0.25*imgT.shape[0]))
    imgMap = np.empty(shape,dtype='uint8')
    imgMap = cv2.resize(imgT,imgMap.shape)
    from scene.scene import Scene
    scene = Scene(imgMap)
    return scene

def run():

    nr = 24

    # list = range(1, 7)
    # list.extend(range(11, 17))
    # list.extend(range(21, 26))
    # list = range(1,16)
    list = [2]

    folder = 7
    # print list
    for i in list:
        edge = None
        filename = 'img/%d/%d.JPG' % (folder, i)
        scene = loadImage(filename)

        FirstCycleFlag = True
        #0.45
        gammas = [0.45]

        for j in gammas:
            # 11
            for n in [5]:
                scene.gauss_kernel = n
                scene.gamma = j
                print (n,j)
                
                edge = scene.getEdges()
                
                f = 'img/results/matching/%d/folder_%d_%d_edge_.jpg' % (folder,folder, i)
                cv2.imwrite(f,edge.map)

                mirror_line = edge.getMirrorLine()
                #
                f = 'img/results/matching/%d/folder_%d_gray.jpg' % (folder,i)
                cv2.imwrite(f,scene.gray)

                # pocdział na obrazy górny i dolny
                view_reflected, view_direct = scene.divide(mirror_line)
                reflected, direct = edge.divide(mirror_line)
                
                # wyrównanie wymiarów (tak żeby oba miały ten sam kształt) wg mniejszego

                up_height = view_reflected.height
                down_height = view_direct.height
                delta = abs(up_height - down_height)
                if up_height > down_height:
                    img_up = view_reflected.view[delta:,:]
                    edge_up = reflected.map[delta:,:]
                    pass
                else:
                    img_down = view_direct.view[:down_height-delta,:]
                    edge_down = direct.map[:down_height-delta:,:]
                    pass

                # GÓRNY OBRAZ
                reflected.findObject()
                #mainBND,mainSqrBnd,contours,objects = findObject(edge.reflected.map,scene.reflected.view)
                myObject = reflected.mainObject
                myObject.markOnCanvas(reflected.view,(255,0,0))

                shape = (reflected.shape[0],reflected.shape[1])

                #znajdź elementy obiektu głównego
                myObject.getStructure()
                #corners,longestContour,lines,left,right,crossing,poly,innerSegments,innerLines,cnt2 = analise(reflected.mainObject.CNT,reflected.contours.copy(),shape)

                #znalezienie markera
                marker_up = features.findMarker(reflected.objects,shape,reflected.map,reflected.view)

                # if marker_up[0] is not None:
                #     xm,ym,wm,hm = cv2.boundingRect(marker_up[0])
                #     markerSqrBnd = (xm,ym,wm,hm)
                # else:
                #     markerSqrBnd = (0,0,0,0)

                #stuff_up = (myObject.corners,reflected.mainObject.sqrBnd,reflected.contours,myObject.longestContour,myObject.left,myObject.right,myObject.lines,myObject.crossing,myObject.poly,myObject.innerSegments,myObject.innerLines,marker_up[0])
                stuff_up = (myObject,marker_up[0])
                img_up = markFeatures(reflected.view,stuff_up)


                # DOLNY OBRAZ
                
                direct.findObject()
                directObject = direct.mainObject
                #directObject.markOnCanvas(direct.view,(255,0,0))

                shape = (edge_down.shape[0],edge_down.shape[1])

                #znajdź elementy obiektu głównego
#                 corners,longestContour,lines,left,right,crossing,poly,innerSegments,innerLines,cnt2 = analise(direct.mainObject.CNT,direct.contours.copy(),shape)
                
                #znajdź elementy obiektu głównego
                directObject.getStructure()
                
                #znalezienie markera
                marker_down = features.findMarker(direct.objects,shape,edge_down,img_down)
                
                # if marker_down[0] is not None:
                #     xm,ym,wm,hm = cv2.boundingRect(marker_down[0])
                #     markerSqrBnd = (xm,ym,wm,hm)
                # else:
                #     markerSqrBnd = (0,0,0,0)

#                 stuff_down = (corners,direct.mainObject.sqrBnd,direct.contours,longestContour,left,right,lines,crossing,poly,innerSegments,innerLines,marker_down[0])
                stuff_down = (directObject,marker_down[0])
                
                img_down = markFeatures(img_down,stuff_down)

                # kalibruj kamere
                ca.calibrate(marker_down,marker_up,scene.getGrayScaleImage(),mirror_line[1],scene.view,direct.shape)


                #zapisz wyniki

                f = 'img/results/matching/%d/folder_%d_%d_cont2_gora_.png' % (folder,folder,i)
                cv2.imwrite(f,img_up,[cv2.IMWRITE_PNG_COMPRESSION,0] )

                f = 'img/results/matching/%d/folder_%d_%d_cont2_dol_.png' % (folder,folder,i)
                cv2.imwrite(f,img_down)


    ch = cv2.waitKey()
    
    
run()