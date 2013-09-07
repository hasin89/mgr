# -*- coding: utf-8 -*-
from cmath import sqrt
from numpy.lib.function_base import average
from func import histogram

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
import func.histogram


def markCounturResult(img,edge):
    vis_filtred = img.copy()

    vis_filtred[edge != 0] = (0, 255, 0)

    return vis_filtred


def track(img, nr, gauss_kernel=11, gammaFactor=0.45):
    """

    :param img:
    :param nr:
    """
    edge_filtred,vis = features.canny(img,gauss_kernel,gammaFactor)
    # edge_filtred,vis = features.adaptiveThreshold(img,gauss_kernel,0.5,-3)

    return edge_filtred


def analise(edge,img=0):
    '''
        znajduje kontury, rogi konturu,
    return (cornerList,mainCNT,cornerCNT,contours_up,longestContour)
    '''

    shape = (edge.shape[0],edge.shape[1])

    contours = features.findContours(edge)

    # objects zawiera obiekty znalezione na podstawie konturu
    objects = features.findObjects(shape,contours)

    # mainCNT zawiera  wielokatna obwiednie bryły
    mainBND = features.findMainObject(objects,shape)

    #prostokatna obwiednia bryły
    #przepisanie z listy na tablice numpy
    mainCNT2 = np.asarray( [[(p[0],p[1]) for p in mainBND[0][:,0]]]  )
    x,y,w,h = cv2.boundingRect(mainCNT2)
    mainSqrBnd = (x,y,w,h)
    cont = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    rectangle = np.asarray([cont])

    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # browse.browse(img)

    # to jest usuwanie konturow nie zwiazanych z obiektem glownym
    toDel = []
    for key,c in contours.iteritems():
        if len(c)>0:
            isinside = cv2.pointPolygonTest(mainBND[0],(c[0][1],c[0][0]),0)
        else:
            isinside = 0
        if isinside != 1:
            contours[key] = []
            pass
        else:
            print key

    #wykrycie wierzchołków
    corners,longestContour, cornerObj = features.findCorners(shape,contours)

    #lista wierzchołkow zwiazanych z bryła i zredukowanych
    cornerObj = features.eliminateSimilarCorners(cornerObj,mainBND,shape)

    # wytypowanie wierzchołków najbardziej lewego i prawego
    corners = np.asarray(cornerObj)
    # na wypadek gdyby nie znalazły się żadne wierzchołki wewnątrz głównego konturu
    if corners.size > 0:
        #znajdz wierzchołek o naniejszej wspolrzednej X
        leftX = min(corners.T[0])
        leftIndex = corners.T.tolist()[0].index(leftX)
        leftY = corners.T[1][leftIndex]
        left = (leftX,leftY)

        #znajdz wierzcholek o najwiekszej wspolrzednj X
        rightX = max(corners.T[0])
        rightIndex = corners.T.tolist()[0].index(rightX)
        rightY = corners.T[1][rightIndex]
        right = (rightX,rightY)
    else:
        left = (0,0)
        right = (shape[1],shape[0])

    tmpbinary = np.zeros(shape,dtype='uint8')
    tmpbinary[:][:] = 0

    for c in longestContour:
        tmpbinary[c] = 1


    rho = 1
    # theta = 0.025
    theta = 0.025
    threshold = 125
    #znaldź linie hougha
    lines = cv2.HoughLines(tmpbinary,rho,theta,threshold)[0]

    crossing = features.getCrossings(lines,edge)


    # cornerList - wierzchołki obiektu
    # cornerCNT - wierzchołki na konturach
    # contours - słownik konturów
    # longestContour - najdłuzszy kontur
    # sqrBND - prostokąt obejmujący obiekt w formacie (x,y,width,height) czyli RAMA
    # left - wierzchołek najbardziej z lewej w ramie
    # right - wierzchołek najbardziej z prawej w ramie
    # lines - proste zwrócone przez algorytm hougha
    # crossing - punkty przeciecia prostych


    return (cornerObj,mainSqrBnd,corners,contours,longestContour,left,right,lines,crossing)


def markFeatures(src,stuff):

    (cornerList,mainBND,cornerCNT,contours,longestContour,left,right,lines,crossing) = stuff

    img = src.copy()
    img[:][:] = (0,0,0)

    # zaznaczenie krawędzi na biało
    img = mark.contours(img,contours)

    #zaznaczenie najdluższego znalezionego konturu na zielono
    img = mark.singleContour(img,longestContour)

    #zaznaczenie wierzchołków na niebiesko
    img = mark.corners(img,cornerList)

    #zaznaczenie interesujących miejsc (obiektu glownego na niebiesko
    img = mark.object(img,mainBND)

    img = mark.point(img,left)
    img = mark.point(img,right)

    #to zaznaczało srodki ciezkosci na biaol
    #img = mark.points(img,centrum)

#zaznaczanie centrum obrazu na żółto
    xc0 = img.shape[0]/2
    yc0 = img.shape[1]/2
    point = (yc0,xc0)
    img = mark.YellowPoint(img,point)

    for p in crossing:
        img = mark.YellowPoint(img,(p[1],p[0]))

    # zazmacz lini hougha
    features.drawLines(lines,img)

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

def run():

    nr = 24

    # list = range(1, 7)
    # list.extend(range(11, 17))
    # list.extend(range(21, 26))
    # list = range(1,16)
    list = [5]

    folder = 4
    # print list
    for i in list:
        edge = None
        filename = 'img/%d/%d.JPG' % (folder, i)
        print(filename)
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        FirstCycleFlag = True
        #0.45
        gammas = [0.45]

        for j in gammas:
            # 11
            for n in [11]:
                gaussKernel = n
                gammaFactor = j
                print (n,j)

                edge = track(img, i,gaussKernel,gammaFactor)

                f = 'img/results/edges/%d/folder_%d_%d_edge_.jpg' % (folder,folder, i)
                # cv2.imwrite(f,edge)

                hy,a,b,mirror_line = countNonZeroRowsY(edge)
                f = 'img/results/contours/%d_histogram_Y.jpg' % (i)
                #
                # cv2.imwrite(f,hy)

                hx,c,d = countNonZeroRowsX(edge)
                # f = 'img/results/histogram/3_avg/%d_histogram_X.jpg' % (i)
                # # cv2.imwrite(f,hx)
                #
                margin = 0
                #
                edge_up = edge[:mirror_line[0],:]
                img_up = img[:mirror_line[0], :]

                edge_down = edge[mirror_line[1]:,:]
                img_down = img[mirror_line[1]:,:]



                stuff_up = analise(edge_up)



                img_up = markFeatures(img_up,stuff_up)


                #
                # linesGeneral = []
                #
                # for (rho, theta) in stuff_up[7]:
                #     # blue for infinite lines (only draw the 5 strongest)
                #     a,b,c = features.convertToGeneral((rho,theta),img_up)
                #     linesGeneral.append((a,b,c))
                #
                #     #test czy te proste generalne sie zgadzaja
                #
                #     #dolna ekranu
                #     y1 = img_up.shape[0]
                #     x1 = int((b*y1+c)/(-a))
                #
                #     #górna krawedź
                #     x2 = int(-c/a)
                #     y2 = 0
                #
                #     # pt1 = (x1,y1)
                #     # pt2 = (x2,y2)
                #     # cv2.circle(img_up, pt1 ,7,(2,255,255,0),3)
                #     # cv2.circle(img_up, pt2 ,7,(2,255,255,0),3)
                #     # cv2.line(img_up, pt1, pt2, (120,255,0), 4)
                #
                #
                #
                #
                #
                # print 'd'


                f = 'img/results/matching/%d/folder_%d_%d_cont2_gora_.png' % (folder,folder,i)
                cv2.imwrite(f,img_up,[cv2.IMWRITE_PNG_COMPRESSION,0] )


                # dolny obraz

                stuff_down = analise(edge_down,img_down)

                img_down = markFeatures(img_down,stuff_down)

                f = 'img/results/matching/%d/folder_%d_%d_cont2_dol_.png' % (folder,folder,i)
                cv2.imwrite(f,img_down)

                im = np.append(img_up,img_down,0)

                height = img_up.shape[0]
                l_d = [l for l in stuff_down[5]]
                r_d = [l for l in stuff_down[6]]

                left_down = (l_d[0],l_d[1]+height)
                right_down = (r_d[0],r_d[1]+height)


                cv2.line(im,stuff_up[5],left_down,(255,255,0),2)
                cv2.line(im,stuff_up[6],right_down,(255,255,0),2)

                # f = 'img/results/contours/parowanie/folder_%d_%d_cont2_all_.png' % (folder,i)
                # cv2.imwrite(f,im)


                # h = cv2.pyrDown(h)
                # FirstCycleFlag = True
                # # browse.browse(edge)
                # if FirstCycleFlag:
                #     edgeSum = np.logical_or(edge,np.zeros(edge.shape))
                #     FirstCycleFlag = False
                # else:
                #     edgeSum = np.logical_or(edgeSum,edge)
                # f = 'img/results/histogram/3_avg/%d_src.jpg' % (i)
                # cv2.imwrite(f, img)
                # f = 'img/results/histogram/3_avg/%d_edge.jpg' % (i)
                # cv2.imwrite(f, edge)

        # edge_gamma = np.array(edgeSum,dtype='uint8')
        # vis = edge_gamma.copy()
        # vis[edge_gamma != 0] = 255
        # f2 = 'img/results/gamma_sum/3/edge_%d_gamma_sum_45_155.jpg' % i
        # cv2.imwrite(f2, vis)
        # browse.browse(vis)
    ch = cv2.waitKey()


run()
# p = []
# for k in range(2,12):
#     p.append((k,2))
# for k in range(3,41):
#     p.append((11,k))
# for k in range(3,12):
#     z = (13-k,40)
#     p.append(z)
# for k in range(3,40):
#     z = (2,42-k)
#     p.append(z)
#
# a = (2,2)
# b = (11,2)
# c = (11,40)
# d = (2,40)
#
# m=(0,20)
# n=(3,2)
# l = features.getLine(a,c)
# dist = features.calcDistances(p)
# cc = features.findCorners([p])
# print cc