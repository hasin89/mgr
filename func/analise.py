# -*- coding: utf-8 -*-
from math import sqrt

__author__ = 'tomek'


import cv2
import numpy as np
import copy


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


def getCrossings(lines,shape,boundaries):
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

    pairs.extend([(linesGeneral[i],linesGeneral[i+2]) for i in range(0,len(linesGeneral)-2)])

    # pairs.extend([(linesGeneral[i],linesGeneral[i+3]) for i in range(0,len(linesGeneral)-3)])


    segments = {}
    vertexes = {}

    for i in range(0,len(linesGeneral)):
        segments[linesGeneral[i]] = obj.Segment()
        segments[linesGeneral[i]].line = linesGeneral[i]

    crossing = []
    good = 0
    j= 0

    #znajdź właściwe przecięcia
    #todo budowa wierzchołków strukturalnych
    for i,(k,l) in enumerate(pairs):

        p = get2LinesCrossing(k,l)
        # sprawdź czy leży wewnątrz ramy
        if p != False:
            isinside = cv2.pointPolygonTest(boundaries,p,0)
            if isinside>0:
                if p != False:
                    crossing.append(p)

                    s1 = segments[k]
                    s1.neibourLines.append(l)
                    s1.points.append(p)
                    segments[k] = s1

                    s2 = segments[l]
                    s2.neibourLines.append(k)
                    s2.points.append(p)
                    segments[l] = s2

                    vertex = obj.Vertex(p)
                    vertex.lines.append(l)
                    vertex.lines.append(k)
                    vertexes[p] = vertex

                    good += 1

            else:
                pass
        if good == len(linesGeneral):
            break

    segmentsList = segments.values()
    poly = obj.Polyline()

    for s in segments.values():
        if len(s.points) > 1:
            vertexes[s.points[0]].neibours[s.line] = s.points[1]
            vertexes[s.points[1]].neibours[s.line] = s.points[0]


    #  nie wiem czy to do czegoś potrzbne jest
    flag = [True for true in segmentsList]
    if (len(segmentsList[0].points)) > 1 & flag[0]:
        poly.segments[0] = segmentsList[0]
        poly.begining = segmentsList[0].points[0]
        poly.ending = segmentsList[0].points[1]
        poly.points.append(segmentsList[0].points[0])
        poly.points.append(segmentsList[0].points[1])
        flag[0] = False

        i = 1
        while(i<len(linesGeneral)):
            s = segmentsList[i]
            points = list(segmentsList[i].points)
            if (poly.ending in points) & flag[i] & (len(segmentsList[i].points) > 1):
                index = points.index(poly.ending)
                if index == 0:
                    index = 1
                else:
                    index = 0
                p = s.points[index]
                poly.ending = p
                poly.points.append(p)
                poly.segments[len(poly.segments)] = segmentsList[i]
                flag[i] = False
                i=0
            else:
                i += 1

    return crossing,poly,vertexes


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


def calcDistances(segment,size,treshold=2):
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

            if dist>treshold:

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

def getMostSeparatedPointsOfLine(points):
    '''
    znajdz najbardziej od siebie odlegle punty
    '''
    max = 0
    for p1 in points:
        for p2 in points:
            if p1 == p2:
                continue
            dist = calcLength(p1, p2)
            if dist>max:
                max = dist
                pair = [p1,p2]
    return pair

def getInnerSegments(otherLines,shape,poly):
    '''
    szukanie odcinka będącego krawędzią wewnętrzeną

    '''
    result = {}
    for l in otherLines:
        l = convertLineToGeneralForm(l,shape)
        min = 100000
        points = list(poly.points)
        del points[-1]
        for p in points:
            dist = calcDistFromLine(p,l)
            if dist < min:
                min = dist
                Pmin1 = p
        del points[points.index(Pmin1)]

        #szukanie drugoego punktu
        min = 100000
        for p in points:
            dist = calcDistFromLine(p,l)
            if dist < min:
                min = dist
                Pmin2 = p

        s = obj.Segment()
        s.setPoints(Pmin1,Pmin2)
        result[l] = s
    # Pmin2 = (0,0)

    return result.values()

def addSegmentsToStructure(innerSegments,vertexes):
    for s in innerSegments:
        for p in s.points:
            vertexes[p].lines.append(s.line)
            index = s.points.index(p)
            if index == 1:
                vertexes[p].neibours[s.line] = s.points[0]
            else:
                vertexes[p].neibours[s.line] = s.points[1]

    pass

def makeFaces(vertexesORG):

    vertexes = copy.deepcopy(vertexesORG)
    line = {}
    i=-1
    for v in vertexes.values():
        start = v.point
        #wszystkie trasy z tego punktu
        routes = []
        #jeśli sa jeszcze jacyś sąsiedzi startu
        while len(vertexes[start].neibours.values())>0:
            # zacznij spisywać trasę
            route = []
            route.append(start)
            point = start

            while len(vertexes[point].neibours.values())>0:

                #pobierz sasiadów
                neibours = vertexes[point].neibours.values()

                #usuń tych co już byli na tej trasie
                for r in route:
                    if r in neibours:

                        del vertexes[point].neibours[vertexes[point].neibours.index(r)]

                #jeżeli już nikt nie został (zatoczyła się pętla)
                if len(neibours) == 0:
                    break

                #wybierz następnika
                point = neibours[0]
                route.append(point)

                pass
                #usuń poprzednika
                # del vertexes[point].neibours[vertexes[point].neibours.values().index(point)]
            routes.append(route)
            pass

            # while start not in point.neibours.values():
            #     for l in line:
            #         line[i].append(neibours[i])
            #
            #     neibours = vertexes[point].neibours.values()
            #     if start in neibours:
            #         break





        pass
    # line[i].append(point)
    # flag = True
    # j=0
    # while(flag):
    #     points = list(vertexes[point].neibours.values())
    #     for l in line[i]:
    #         if l in points:
    #             del points[points.index(l)]
    #     if len(points)>0:
    #         point = points[0]
    #         line[i].append(point)
    #     else:
    #         break



    return line


def tryMatch(corners,left,right):
    min = left[0]
    max = right[0]
    Xs = []

    for c in corners:
        x = c[0]-min
        x /= float(max-min)
        Xs.append((x,(c[0],c[1])))
    Xs.sort()
    return Xs


# [(0.0, 962), (0.077071290944123308, 569), (0.39306358381502893, 354), (1.0, 775)]
# [(0.0, 349), (0.069602272727272721, 1092), (0.42329545454545453, 320), (1.0, 648)]