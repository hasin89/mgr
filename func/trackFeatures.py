# -*- coding: utf-8 -*-
from math import sqrt

__author__ = 'tomek'
#/usr/bin/env python

'''
This sample demonstrates Canny edge detection.

Usage:
  edge.py [<video source>]

  Trackbars control edge thresholds.

'''

import cv2
import browse
import numpy as np
import func.gamma as gamma

def canny(img, gauss_kernel=11, gammaFactor=0.45):
    """

    :param img:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = gauss_kernel
    gray_filtred = gamma.correction(gray,gammaFactor)
    gray_filtred = cv2.GaussianBlur(gray_filtred, (k, k), 0)

    thrs1 = 20
    kernel = 3
    ratio = 3
    edge_filtred = cv2.Canny(gray_filtred, thrs1, thrs1 * ratio, kernel)

    # zaznaczenie wyniku wyznaczania krawedzi

    vis_filtred = img.copy()
    # vis_filtred /= 2
    vis_filtred[edge_filtred != 0] = (0, 255, 0)

    return edge_filtred,vis_filtred


def threshold(img, gauss_kernel=11, gammaFactor=0.45, threshold = 50):


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = gauss_kernel
    gray_filtred = gamma.correction(gray,gammaFactor)
    gray_filtred = cv2.GaussianBlur(gray_filtred, (k, k), 0)

    edge_filtred = cv2.threshold(gray_filtred,threshold,maxval=1,type=cv2.THRESH_BINARY)[1]

    vis_filtred = img.copy()
    vis_filtred[edge_filtred != 0] = (0, 255, 0)

    return edge_filtred,vis_filtred


def adaptiveThreshold(img, gauss_kernel=11, gammaFactor=0.45, C = -3):


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = gauss_kernel
    gray_filtred = gamma.correction(gray,gammaFactor)
    gray_filtred = cv2.GaussianBlur(gray_filtred, (k, k), 0)

    Cval = C
    edge_filtred = cv2.adaptiveThreshold(gray_filtred,maxValue=1,adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=11,C=Cval)

    vis_filtred = img.copy()
    vis_filtred[edge_filtred != 0] = (0, 255, 0)

    return edge_filtred,vis_filtred


def gammaSum(img,gaussKernel=11):

    FirstCycleFlag = True
    gammas = [0.25,4]

    for j in gammas:
        gammaFactor = j
        edge = canny(img,gaussKernel,gammaFactor)[0]
        if FirstCycleFlag:
            edgeSum = np.logical_or(edge,np.zeros(edge.shape))
            FirstCycleFlag = False
        else:
            edgeSum = np.logical_or(edgeSum,edge)

    edge_gamma = np.array(edgeSum,dtype='uint8')
    vis = edge_gamma.copy()
    vis[edge_gamma != 0] = 255
    return vis


def findContours(edge):
    '''
    łaczy punkty w kontur
    '''

    contours = {0:[]}
    i = 0
    flag = True

    nonzeros = np.nonzero(edge)
    pointX = nonzeros[1][0]
    pointY = nonzeros[0][0]
    contours[0].append((pointY,pointX))

    while flag == True:
        neibour = getNeibours(edge,pointX,pointY)
        #jeżeli znaleziono sąsiada tododaj go do konturu
        if neibour != False:
            edge[pointY,pointX] = 0
            pointX = neibour[0]
            pointY = neibour[1]
            contours[i].append((pointY,pointX))

        #jeżeli nie znaleziono sąsiada
        else:
            #poszukaj w otoczeniu
            edge[pointY,pointX] = 0
            near, dist = searchNearBy(edge,pointX,pointY)
            # near = np.asarray([])
            # dist = -1

            #jezeli znaleziono punkt w otoczeniu
            if near.size>0 :
                pointX = near[0]
                pointY = near[1]

            #jezeli nie znaleziono w otoczeniu żadnego punktu to znajdź pierwszy niezerowy
            else:
                nonzeros = np.nonzero(edge)
                if nonzeros[0].size > 0:
                    pointX = nonzeros[1][0]
                    pointY = nonzeros[0][0]
                    # cv2.circle(nimg, (pointX,pointY),10,(255,255,255),1)
                    # nimg[pointY,pointX] = (255,0,0)

                # jeżeli nie ma żadnych niezerowych to zakończ algorytm szukania krawędzi
                else:
                    flag = False

            # jeżeli punktu nie był w otoczeniu w odległości 5 to znaczy że to nowy kontur
            if dist != 5:
                i+=1
                # jeżeli to nie koniec algorytmu to zacznij nową krawędź
                if flag != False:
                    contours[i]=[]

            # jeżeli to nie koniec algorytmu to dodaj punkt znaleziony dowolną meodą do bierzacego konturu
            if flag != False:
                contours[i].append((pointY,pointX))

    # sklejenie konturow ktore powinny byc jednak razem
    neibours = []
    neibours = getContourNeibours(contours,contours[0][0])
    if len(neibours)>0:
        for n in neibours:
            if n[0] != 0:
                contours[0].extend(contours[n[0]])
                contours[n[0]] = []

    print "ilosc"
    print len(contours)

    return contours


def getNeibours(edge,x,y):

    yMax,xMax = edge.shape

    check = {}

    check['a'] = (x+1,y)
    check['c'] = (x-1,y)
    check['e'] = (x+1,y+1)
    check['g'] = (x-1,y-1)

    check['b'] = (x, y+1)
    check['d'] = (x,y-1)
    check['f'] = (x-1,y+1)
    check['h'] = (x+1,y-1)


    if x == 0:
        del check['c']
        del check['f']
        del check['g']

    elif x == xMax-1:
        del check['a']
        del check['e']
        del check['h']

    if y == 0:
        del check['d']
        if 'h' in check.keys():
            del check['h']
        if 'g' in check.keys():
            del check['g']

    elif y == yMax-1:
        del check['b']
        if 'e' in check.keys():
            del check['e']
        if 'f' in check.keys():
            del check['f']

    for value in check.itervalues():
        if edge[value[1],value[0]] == 255:
            return (value[0],value[1])

    return False


def getContourNeibours(contours,(x,y)):


    check = {}

    check['a'] = (x+1,y)
    check['c'] = (x-1,y)
    check['e'] = (x+1,y+1)
    check['g'] = (x-1,y-1)

    check['b'] = (x, y+1)
    check['d'] = (x,y-1)
    check['f'] = (x-1,y+1)
    check['h'] = (x+1,y-1)

    neibours = []
    for k,v in contours.items():
        for point in check.itervalues():
            try:
                index = v.index(point)
            except ValueError:
                index = -1
            if index != -1:
                neibours.append((k,index))

    return neibours


def searchNearBy(edge,x,y):
    print 'search'
    sp = edge.T.shape

    maskSizes = range(5,19,2)
    for i in maskSizes:
        mask = generateMask(i)
        points = np.asarray((x,y))+mask

        #filtowanie z poza granic
        xx = points[:,0]
        yy =points[:,1]
        XoverflowIndex = np.where(xx>sp[0]-1)
        YoverflowIndex = np.where(yy>sp[1]-1)
        wrongIndexes = np.union1d(XoverflowIndex[0],YoverflowIndex[0])
        points= np.delete(points,wrongIndexes,0)

        p = [edge[points[k][1],points[k][0]] for k in range(len(points))]
        non = np.nonzero(p)[0]
        if non.size>0:
            no = np.nonzero(p)[0][0]
            return points[no] , i

    return np.asarray([]),False


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

def getCrossings(lines,edge):
    '''
    zwraca punkty bedące przecięciami podanych prostych

    lines - lista prostych będących krotkami [ (a,b,c) , ... ]
    edge - płótno

    return [ (x,y) ] - lista puntków będacych przecieciami
    '''
    linesGeneral = []

    for (rho, theta) in lines:
        # blue for infinite lines (only draw the 5 strongest)
        a,b,c = convertLineToGeneralForm((rho,theta),edge)
        linesGeneral.append((a,b,c))

    pairs = [(linesGeneral[i],linesGeneral[i+1]) for i in range(0,len(linesGeneral)-1)]
    pairs.append((linesGeneral[-1],linesGeneral[0]))

    crossing = []
    for k,l in pairs:
        p = get2LinesCrossing(k,l)
        if p != False:
            crossing.append(p)

    return crossing

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

def generateMask(maskSize):

    n = (maskSize-1)/2
    mask = []
    x=n
    y=n-1
    while x<n+1:
        while y<n:
            while x>-n:
                while y>-n:
                    mask.append((x,y))
                    y-=1
                mask.append((x,y))
                x-=1
            mask.append((x,y))
            y+=1
        mask.append((x,y))
        x+=1

    return mask


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


def findCornersOnContour(contour,size):

    if len(contour)>size:
        indexes = []
        dist = calcDistances(contour,size)

        for d in dist.iterkeys():
            segment1 = dist[d]
            MaxValue = max(np.asarray(segment1)[:,1])
            index = np.where(segment1 == MaxValue)[0]
            indexes.append(segment1[index[0]][0])
        return indexes
    else:
        return []


def drawLines(lines,img):

    # dla kolorowych obrazow sa 3 wymiary , 3 jest zbędny nam potem
    m,n,w = img.shape

    for (rho, theta) in lines[:20]:
        # blue for infinite lines (only draw the 5 strongest)
        x0 = np.cos(theta)*rho
        y0 = np.sin(theta)*rho
        pt1 = ( int(x0 + (m+n)*(-np.sin(theta))), int(y0 + (m+n)*np.cos(theta)) )
        pt2 = ( int(x0 - (m+n)*(-np.sin(theta))), int(y0 - (m+n)*np.cos(theta)) )
        cv2.line(img, pt1, pt2, (128,0,128), 2)
    return img

def convertLineToGeneralForm((rho,theta),edge):
    '''
    konwertuje prosta we wspórzednych biegunowych (rho,theta)
    do postaci ogólnej Ax+By+c = 0

    rho - odległość prostej od (0,0)
    theta - kąt prostej

    return (a,b,c)
    '''

    x0 = np.cos(theta)*rho
    y0 = np.sin(theta)*rho

    m,n = edge.shape
    pt1 = ( int(x0 + (m+n)*(-np.sin(theta))), int(y0 + (m+n)*np.cos(theta)) )
    pt2 = ( int(x0 - (m+n)*(-np.sin(theta))), int(y0 - (m+n)*np.cos(theta)) )


    # cv2.circle(img, pt1 ,7,(2,255,255,0),3)
    # cv2.circle(img, pt2 ,7,(2,255,255,0),3)
    # cv2.line(img, pt1, pt2, (120,255,0), 4)

    line= getLine((pt1[1],pt1[0]),(pt2[1],pt2[0]))

    return line


def eliminateSimilarCorners_old(corners,nimg,border=35):

    img2 = cv2.copyMakeBorder(nimg,border,border,border,border,cv2.BORDER_CONSTANT,None,0)

    #słownik blixkich sobie wierzchołków
    semi = {}

    # znalezienie bliskich sobie wierzchołków
    for (x,y) in corners:
        img3 = img2[y:y+border*2,x:x+2*border]
        non = np.nonzero(img3)
        if len(non[0])>1:
            for k in range(len(non[0])):
                vecY = non[0][k]-border
                vecX = non[1][k]-border
                if (vecX != 0) & (vecY != 0):
                    new = (x+vecX, y+vecY)
                    semi[(x,y)] = []
                    semi[(x,y)].append((x,y))
                    semi[(x,y)].append(new)
                    try:
                        del corners[corners.index((x,y))]
                    except ValueError:
                        pass
                    try:
                        del corners[corners.index(new)]
                    except ValueError:
                        pass

    # obliczanie średniej bliskich sobie wierzchołków
    for list in semi.itervalues():
        xx = np.asarray(list)[:,0]
        x_ = int(np.average(xx))
        yy = np.asarray(list)[:,1]
        y_ = int(np.average(yy))

        # dodanie do listy globalnej wierzchołków
        corners.append((x_,y_))
        pass
    return corners


def eliminateSimilarCorners(corners,mainCnt,shape,border=35):
    '''
    eliminuje wierzchołki ktore prawdopodobnie sa blisko siebie
    '''

    # płótno pod wierzchołki
    nimg = np.zeros(shape,dtype='uint8')
    nimg[:][:] = 0

    cornersInside = []
    distances = {}

    #wybranie tylko tych wierzchołkow ktore leza wewnatrz obwiedni
    for pt in corners:

        #sprawdź czy leży w konturze i w jakiej odległości
        result = cv2.pointPolygonTest(mainCnt[0],pt,1)

        #jeżeli leżą wewnątrz
        if result>=0:

            #zpisz na listę
            cornersInside.append(pt)

            #zapisz odległość
            distances[pt] = result

            #zaznacz na płótnie
            nimg[pt[1],pt[0]] = 255

    # zabezpieczenie przed przelewaniem
    img2 = cv2.copyMakeBorder(nimg,border,border,border,border,cv2.BORDER_CONSTANT,None,0)

    #słownik blixkich sobie wierzchołków
    semi = {}

    # znalezienie bliskich sobie wierzchołków
    for (x,y) in cornersInside:

        #wyodrębnij obszar
        img3 = img2[y:y+border*2,x:x+2*border]

        #znajdź punkty na obszarze
        non = np.nonzero(img3)


        #jezeli jest więcej niż jeden
        if len(non[0])>1:

            semi[(x,y)] = []
            semi[(x,y)].append((x,y))

            for k in range(len(non[0])):

                #znajdź wektor między punktem a środkiem obszaru
                vecY = non[0][k]-border
                vecX = non[1][k]-border

                #jezeli jest to wektor niezerowy to mamy punkt
                if (vecX != 0) | (vecY != 0):
                    new = (x+vecX, y+vecY)
                    semi[(x,y)].append(new)
                    try:
                        del cornersInside[cornersInside.index((x,y))]
                    except ValueError:
                        print 'error a'
                        print (x,y)
                        pass
                    try:
                        del cornersInside[cornersInside.index(new)]
                    except ValueError:
                        print 'error new'
                        print (x,y)
                        pass

    # wybranie wierzchołków bliższych konturowi zewnętrzenmu
    for list in semi.itervalues():
        dist = [distances[li] for li in list]
        minIndex = dist.index(min(dist))
        (x_,y_) = list[minIndex]

        # dodanie do listy globalnej wierzchołków
        cornersInside.append((x_,y_))
    return cornersInside


def findCorners(shape,contours):

    # lista z podziełem na kontury
    cornerCNT = {}

    #lista bez podziału na kontury
    cornerList = []

    longestContour = []

    #dla każdego znalezionego konturu
    for cindex in range(len(contours)):
        cornerCNT[cindex] = []

        # szukanie najdłuższego - obiedni
        cnt_len = len(contours[cindex])
        if cnt_len>len(longestContour):
            longestContour = contours[cindex]

        indexes = findCornersOnContour(contours[cindex],16)

        # zaznacz wszsytkie znalezione wierzchołki
        for id in indexes:
            (y,x) = contours[cindex][id]

            # dodaj do globalnej listy ze wskazaniem konturu
            cornerCNT[cindex].append((x,y))

            # dodaj do listy bez wskazania konturu
            cornerList.append((x,y))

        # usunięcie podobnych punktów na konturze
        # cornerCNT[cindex] = eliminateSimilarCorners(cornerCNT[cindex],nimg,border=35)

    return cornerCNT,longestContour,cornerList


def findObjects(shape,contours):
    '''
     znajduje obiekty na podstawie konturów zmalezionych (łaczy poblisike kontury w jeden obiekt
    '''

    rectangles = []
    tmpbinary = np.zeros(shape,dtype='uint8')
    tmpbinary[:][:] = 0

    #zrób obramowania do każdego konturu
    for c in contours.itervalues():
        if len(c)>0:
            points = np.asarray([c])
            y,x,h,w = cv2.boundingRect(points)

            # powiększ obszar zainteresowania jeśli to możliwe i potrzebna
            # y = y-5;
            # x= x-5
            # w = w+10
            # h = h+10

            cont = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
            rectangle = np.asarray([cont])
            rectangles.append((rectangle))

    # narysuj białe obszary - w ten obszary stykające się lub nakładaące zleją sie w jeden obszar

    margin = 10

    for r in rectangles:
        A = (r[0][0][0]-margin,r[0][0][1]-margin)
        B = (r[0][2][0]+margin,r[0][2][1]+margin)
        cv2.rectangle(tmpbinary,A,B,255,-1)
        # cv2.drawContours(tmpbinary,r,-1,255,-1)

    #znajdź kontury wśród białych prostokątów na czarnym tle
    cntTMP, h = cv2.findContours(tmpbinary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return cntTMP


def findMainObject(objectsCNT,shape):
    '''
     znajduje obiekt najbardziej po srodku plaszczyzny (bo to nie beda krawedzie lustra)
    '''

    #srodek obrazu
    yc0 = shape[0]/2
    xc0 = shape[1]/2

    min_index = -1
    min_cost = shape[0]*shape[1]

    for n,c in enumerate(objectsCNT):
        moments = cv2.moments(c)
        # policzenie srodkow ciezkosci figur
        yc = int(moments['m01']/moments['m00'])
        xc = int(moments['m10']/moments['m00'])

        #odległosc od srodka
        dx = xc0-xc
        dy = yc0-yc
        cost = sqrt(pow(dx,2)+pow(dy,2))

        if cost.real < min_cost:
            min_cost = cost.real
            min_index = n
    mainCNT = [objectsCNT[min_index]]

    return mainCNT