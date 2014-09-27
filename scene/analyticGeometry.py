# -*- coding: utf-8 -*-
from math import sqrt

__author__ = 'tomek'

import numpy as np


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


def getLine(p1,p2,switch = 1):
    '''
    zwraca parametry prostej Ax+By+C na podstawie podanych dwóch punktów
    return (a,b,c)
    '''
    line = []

    if switch == 1:
        (y1,x1) = p1
        (y2,x2) = p2
    else:
        (x1,y1) = p1
        (x2,y2) = p2

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



    F = crossProduct((x1,y1,1),(x2,y2,1))
    E = (a,b,c)
    return (a,b,c)


def get2LinesCrossing((a1,b1,c1),(a2,b2,c2)):
    '''
    zwraca wspołrzedne przecięcia siędwóch prostych podanych parametrami postaci obólnej
    (a1,b1,c1) i(a2,b2,c2)

    return p = (x,y)
    '''

    # reczny iloczyn wektorowy kxl = (x,y,z)

    # z
    # Wab = a1 * b2 - a2 * b1
    #
    # # x
    # Wbc = b1 * c2 - b2 * c1
    #
    # # y
    # Wca = c1 * a2 - c2 * a1

    k = (a1,b1,c1)
    l = (a2,b2,c2)
    (x,y,z) = crossProduct(k,l)

    if z != 0:
        p = (int(x/z),int(y/z))
        return p
    else:
        return False


def crossProduct(k,l):
    '''
    iloczyn wektorowy do krótkich wektorów
    '''
    a1,b1,c1 = k
    a2,b2,c2 = l

    z = a1 * b2 - a2 * b1

    x = b1 * c2 - b2 * c1

    y = c1 * a2 - c2 * a1

    return (x,y,z)


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
                (a,b,c) = getLine(segment[i],segment[i+size],1)

            # jeżeli index jest większy od największego indexu to doklej element z poczatku
            else:
                index = abs(len(segment)-i-size)
                (a,b,c) = getLine(segment[i],segment[index],1)

            # oblicz odległość punktu środkowego od prostej

            # jeśli index mieści się w zakresie indexów
            if i+size/2 < len(segment):
                dist = calcDistFromLine((segment[i+size/2][1],segment[i+size/2][0]),(a,b,c))

            # jeżeli index jest większy od największego indexu to weź element z poczatku
            else:
                index = abs(len(segment)-i-size/2)
                dist = calcDistFromLine((segment[index][1],segment[index][0]),(a,b,c))


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
    x = point[0]
    y = point[1]

    a = line[0]
    b = line[1]
    c = line[2]

    dist = abs(a*x+b*y+c)
    dist = dist/(1.0*sqrt(pow(a,2)+pow(b,2)))

    return dist


def calcLength(p1,p2):
    '''
    liczy odległość między punktami
    '''
    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]

    dist = sqrt(pow(x2-x1,2)+pow(y2-y1,2))
    return dist


# [(0.0, 962), (0.077071290944123308, 569), (0.39306358381502893, 354), (1.0, 775)]
# [(0.0, 349), (0.069602272727272721, 1092), (0.42329545454545453, 320), (1.0, 648)]