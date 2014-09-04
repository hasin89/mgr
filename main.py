# -*- coding: utf-8 -*-
from cmath import sqrt
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
import func.objects as obj
import func.analise as an
import func.calibrate as ca
import func.histogram


def markCounturResult(img,edge):
    vis_filtred = img.copy()

    vis_filtred[edge != 0] = (0, 255, 0)

    return vis_filtred

# default gauss_kernel = 11
# default gamma factor = 0.45


def analise(mainBND,contours,shape,linesThres=25):

    # to jest usuwanie konturow nie zwiazanych z obiektem glownym
    contours2 = features.filterContours(contours,mainBND)

    #wykrycie wierzchołków związanych z konturem
    cornersCNT,longestContour, corners = features.findCorners(shape,contours2)

    #wykrycie lini Hougha na podstawie najdłuższego konturu
    lines = features.findLines(longestContour,shape,linesThres)

    #lista wierzchołkow zwiazanych z bryła i zredukowanych
    corners = features.eliminateSimilarCorners(corners,mainBND,shape)
    # av

    if lines.__class__.__name__ == 'list':
        crossing,poly,vertexes = an.getCrossings(lines,shape,mainBND)

    # wytypowanie wierzchołków najbardziej lewego i prawego
    # corners = np.asarray(cornerObj)
    # left,right = an.getMostLeftAndRightCorner(np.asarray(corners),shape)
        left,right = an.getMostLeftAndRightCorner(np.asarray(crossing),shape)

        an.tryMatch(crossing,left,right)

        lines, innerLines = features.findInnerLines(contours,longestContour,shape,lines)

        innerSegments = an.getInnerSegments(innerLines,shape,poly)

        an.addSegmentsToStructure(innerSegments,vertexes)
    else:
        innerLines = False
        innerSegments = False
        crossing = False
        left = right = poly = False

            # an.makeFaces(vertexes)

    # cornerList - wierzchołki obiektu
    # cornerCNT - wierzchołki na konturach
    # contours - słownik konturów
    # longestContour - najdłuzszy kontur
    # sqrBND - prostokąt obejmujący obiekt w formacie (x,y,width,height) czyli RAMA
    # left - wierzchołek najbardziej z lewej w ramie
    # right - wierzchołek najbardziej z prawej w ramie
    # lines - proste zwrócone przez algorytm hougha
    # crossing - punkty przeciecia prostych

    return corners,longestContour,lines,left,right,crossing,poly,innerSegments,innerLines,contours2


def findObject(edge,img=0):
    '''
        znajduje kontury, rogi konturu,
    return (mainBND,mainSqrBnd,contours,objects)
    '''

    shape = (edge.shape[0],edge.shape[1])

    contours = features.findContours(edge)

    # objects zawiera obiekty znalezione na podstawie konturu
    objects = features.findObjects(shape,contours)

    for tmpobj in objects:
        for i in range(0,len(tmpobj)-1):
            # mark.drawSegment(img,(tmpobj[i][0][0],tmpobj[i][0][1]) ,(tmpobj[i+1][0][0],tmpobj[i+1][0][1]))
            pass


    # obiekt główny
    mainBND = features.findMainObject(objects,shape,img)
    for i in range(0,len(mainBND)-1):
            mark.drawMain(img,(mainBND[i][0][0],mainBND[i][0][1]) ,(mainBND[i+1][0][0],mainBND[i+1][0][1]))

    x,y,w,h = cv2.boundingRect(mainBND)
    # mainSqrBnd zawiera  wielokatna obwiednie bryły
    mainSqrBnd = (x,y,w,h)

    return mainBND,mainSqrBnd,contours,objects


def markFeatures(src,stuff):

    (cornerList,mainBND,contours,longestContour,left,right,lines,crossing,poly,innerSegments,innerLines,markerSqrBnd) = stuff

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
                scene.divide(mirror_line)
                edge.divide()
                
                # wyrównanie wymiarów (tak żeby oba miały ten sam kształt) wg mniejszego

                up_height = scene.reflected.height
                down_height = scene.direct.height
                delta = abs(up_height - down_height)
                if up_height > down_height:
                    img_up = scene.reflected.image[delta:,:]
                    edge_up = edge.reflected[delta:,:]
                    pass
                else:
                    img_down = scene.direct.image[:down_height-delta,:]
                    edge_down = edge.direct[:down_height-delta:,:]
                    pass

                # GÓRNY OBRAZ

                mainBND,mainSqrBnd,contours,objects = findObject(edge.reflected,scene.reflected.image)

                shape = (edge.reflected.shape[0],edge.reflected.shape[1])

                #znajdź elementy obiektu głównego
                corners,longestContour,lines,left,right,crossing,poly,innerSegments,innerLines,cnt2 = analise(mainBND,contours.copy(),shape)

                #znalezienie markera
                marker_up = features.findMarker(objects,shape,edge.reflected,scene.reflected.image)

                # if marker_up[0] is not None:
                #     xm,ym,wm,hm = cv2.boundingRect(marker_up[0])
                #     markerSqrBnd = (xm,ym,wm,hm)
                # else:
                #     markerSqrBnd = (0,0,0,0)

                stuff_up = (corners,mainSqrBnd,contours,longestContour,left,right,lines,crossing,poly,innerSegments,innerLines,marker_up[0])

                img_up = markFeatures(scene.reflected.image,stuff_up)


                # DOLNY OBRAZ

                mainBND,mainSqrBnd,contours,objects = findObject(edge_down,img_down)

                shape = (edge_down.shape[0],edge_down.shape[1])

                #znajdź elementy obiektu głównego
                corners,longestContour,lines,left,right,crossing,poly,innerSegments,innerLines,cnt2 = analise(mainBND,contours.copy(),shape)

                #znalezienie markera
                marker_down = features.findMarker(objects,shape,edge_down,img_down)

                # if marker_down[0] is not None:
                #     xm,ym,wm,hm = cv2.boundingRect(marker_down[0])
                #     markerSqrBnd = (xm,ym,wm,hm)
                # else:
                #     markerSqrBnd = (0,0,0,0)

                stuff_down = (corners,mainSqrBnd,contours,longestContour,left,right,lines,crossing,poly,innerSegments,innerLines,marker_down[0])

                img_down = markFeatures(img_down,stuff_down)

                # kalibruj kamere
                ca.calibrate(marker_down,marker_up,scene.getGrayScaleImage(),mirror_line[1],scene.image,edge_down.shape)


                #zapisz wyniki

                f = 'img/results/matching/%d/folder_%d_%d_cont2_gora_.png' % (folder,folder,i)
                cv2.imwrite(f,img_up,[cv2.IMWRITE_PNG_COMPRESSION,0] )

                f = 'img/results/matching/%d/folder_%d_%d_cont2_dol_.png' % (folder,folder,i)
                cv2.imwrite(f,img_down)

                # im = np.append(img_up,img_down,0)
                #
                # height = img_up.shape[0]
                # l_d = [l for l in stuff_down[5]]
                # r_d = [l for l in stuff_down[6]]

                # left_down = (l_d[0],l_d[1]+height)
                # right_down = (r_d[0],r_d[1]+height)
                #
                #
                # cv2.line(im,stuff_up[5],left_down,(255,255,0),2)
                # cv2.line(im,stuff_up[6],right_down,(255,255,0),2)

                # f = 'img/results/contours/parowanie/folder_%d_%d_cont2_all_.png' % (folder,i)
                # cv2.imwrite(f,im)


    ch = cv2.waitKey()
    
 


run()