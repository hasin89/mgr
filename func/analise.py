# -*- coding: utf-8 -*-
from math import sqrt

__author__ = 'tomek'

import numpy as np
import func.objects as obj


def getMostLeftAndRightCorner(corners,shape):
    '''
    zwraca wierchołki o najmniejszej i największej współrzędnej X

    corners - ndarray - wierzchołki (x,y)
    shape - tuple (h,w)
    '''

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

    return left,right


def getCrossings(lines,shape):
    '''
    zwraca punkty bedące przecięciami podanych prostych

    lines - lista prostych będących krotkami [ (a,b,c) , ... ]
    edge - płótno

    return [ (x,y) ] - lista puntków będacych przecieciami
    '''
    linesGeneral = []

    for (rho, theta) in lines:
        # blue for infinite lines (only draw the 5 strongest)
        a,b,c = convertLineToGeneralForm((rho,theta),shape)
        linesGeneral.append((a,b,c))

    pairs = [(linesGeneral[i],linesGeneral[i+1]) for i in range(0,len(linesGeneral)-1)]
    pairs.append((linesGeneral[-1],linesGeneral[0]))

    segments = {}
    for i in range(0,len(linesGeneral)):
        segments[i] = obj.Segment()

    crossing = []
    for i,(k,l) in enumerate(pairs):

        p = get2LinesCrossing(k,l)
        if p != False:
            crossing.append(p)

            s1 = segments[i]
            s1.line = k
            s1.neibourLines.append(l)
            s1.points.append(p)
            segments[i] = s1

            if i+1 < len(linesGeneral):
                s2 = segments[i+1]
                s2.neibourLines.append(k)
                s2.points.append(p)
                segments[i+1] = s2
            else:
                s2 = segments[0]
                s2.neibourLines.append(k)
                s2.points.append(p)
                segments[0] = s2

    totalSegments = segments.copy()
    poly = obj.Polyline()
    if len(segments[0].points) > 1:
        poly.segments[0] = segments[0]
        poly.begining = segments[0].points[0]
        poly.ending = segments[0].points[1]
        poly.points.append(segments[0].points[0])
        poly.points.append(segments[0].points[1])
        segments[0].points = []

        i = 1
        while(i<len(linesGeneral)):
            s = segments[i]
            if (poly.ending in s.points) & (len(s.points)>1):
                index = s.points.index(poly.ending)
                if index == 0:
                    index = 1
                else:
                    index = 0
                p = s.points[index]
                poly.ending = p
                poly.points.append(p)
                poly.segments[len(poly.segments)] = segments[i]
                segments[i].points = []
                i=0
            else:
                i += 1




    return crossing,poly


def convertLineToGeneralForm((rho,theta),shape):
    '''
    konwertuje prosta we wspórzednych biegunowych (rho,theta)
    do postaci ogólnej Ax+By+c = 0

    rho - odległość prostej od (0,0)
    theta - kąt prostej
    shape - (h,w) - kszatałt płótna

    return (a,b,c)
    '''

    x0 = np.cos(theta)*rho
    y0 = np.sin(theta)*rho

    m,n = shape
    pt1 = ( int(x0 + (m+n)*(-np.sin(theta))), int(y0 + (m+n)*np.cos(theta)) )
    pt2 = ( int(x0 - (m+n)*(-np.sin(theta))), int(y0 - (m+n)*np.cos(theta)) )


    # cv2.circle(img, pt1 ,7,(2,255,255,0),3)
    # cv2.circle(img, pt2 ,7,(2,255,255,0),3)
    # cv2.line(img, pt1, pt2, (120,255,0), 4)

    line= getLine((pt1[1],pt1[0]),(pt2[1],pt2[0]))

    return line


def getLine((y1,x1),(y2,x2)):
    '''
    zwraca parametry prostej Ax+By+C na podstawie podanych dwóch punktów
    return (a,b,c)
    '''
    line = []

    dx = x2-x1
    dy = y2-y1
    if dx != 0:
        a = dy*1.0/dx
        c = y1 - a * x1
        b = -1
    else:
        a = -1
        b = 0
        c = x1

    return (a,b,c)


def get2LinesCrossing((a1,b1,c1),(a2,b2,c2)):
    '''
    zwraca wspołrzedne przecięcia siędwóch prostych podanych parametrami postaci obólnej
    (a1,b1,c1) i(a2,b2,c2)

    return p = (x,y)
    '''
    Wab = a1*b2 - a2*b1
    Wbc = b1 * c2 - b2 * c1
    Wca = c1 * a2 - c2 * a1

    if Wab != 0:
        p = (int(Wca/Wab),int(Wbc/Wab))
        return p
    else:
        return False


def calcDistances(segment,size):
    '''
        liczy odległość punktów odcinka od odcinka z
    '''

    # lista punktów leżących w okolicach wierzchołków
    ds = {}

    # j nie ma fizycznego znaczenia, iteruje docinek po mniejszych składowych
    j=0
    ds[j] = []

    # iteracja po wszystkich elementach odcinka
    for i in range(len(segment)):

            # oblicz współrzędne prostej

            # jeśli index mieści się w zakresie indexów odcinka
            if i+size < len(segment):
                (a,b,c) = getLine(segment[i],segment[i+size])

            # jeżeli index jest większy od największego indexu to doklej element z poczatku
            else:
                index = abs(len(segment)-i-size)
                (a,b,c) = getLine(segment[i],segment[index])

            # oblicz odległość punktu środkowego od prostej

            # jeśli index mieści się w zakresie indexów
            if i+size/2 < len(segment):
                dist = calcDistFromLine(segment[i+size/2],(a,b,c))

            # jeżeli index jest większy od największego indexu to weź element z poczatku
            else:
                index = abs(len(segment)-i-size/2)
                dist = calcDistFromLine(segment[index],(a,b,c))


            # jesli dystans jest większy niż 2

            if dist>2:

                #zapisz index punktu oraz jego odległość

                # jeśli index mieści się w zakresie indexów
                if i+size/2 <len(segment):
                    z = (i+size/2,dist)

                # jeżeli index jest większy od największego indexu to doklej elementy z poczatku
                else:
                    index = abs(len(segment)-i-size/2)
                    z = (index,dist)

                #zapisz punkt posiadający odległość do listy punktów wierzchołkowych
                ds[j].append(z)

            # jeśli dystanc punktu od prostej jest mały  to zwiększ licznik i usuń poprzedni element
            else:
                j+=1
                if len(ds[j-1]) == 0:
                    del ds[j-1]
                ds[j] = []
    del ds[j]
    return ds


def calcDistFromLine(point,line):
    '''
    liczy odleglosc punktu (x,y) od prostej (a,b,c)
    '''
    x = point[1]
    y = point[0]

    a = line[0]
    b = line[1]
    c = line[2]

    dist = abs(a*x+b*y+c)
    dist = dist/(1.0*sqrt(pow(a,2)+pow(b,2)))

    return dist


def calcLength(p1,p2):
    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]

    dist = sqrt(pow(x2-x1,2)+pow(y2-y1,2))
    return dist