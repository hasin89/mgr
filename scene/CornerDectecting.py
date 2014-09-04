# -*- coding: utf-8 -*-
from math import sqrt
from math import cos
from collections import Counter
from numpy.core.defchararray import array
from func import browse
from func import trackFeatures

__author__ = 'tomek'
#/usr/bin/env python

'''

'''

import cv2
import numpy as np
import func.analise as an


class CornerDetector():
    
    
    def __init__(self,contours):
        self.contours = contours
        self.shape = self.edge.shape
    
    
    def findCornersOnContour(self,contour,size):
    
        if len(contour)>size:
            indexes = []
            dist = an.calcDistances(contour,size)
    
            for d in dist.iterkeys():
                segment1 = dist[d]
                MaxValue = max(np.asarray(segment1)[:,1])
                index = np.where(segment1 == MaxValue)[0]
                indexes.append(segment1[index[0]][0])
            return indexes
        else:
            return []
    
      
    def eliminateSimilarCorners(self,corners,mainCnt,shape,border=35):
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
            result = cv2.pointPolygonTest(mainCnt,pt,1)
    
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
                    if (vecX != 0) or (vecY != 0):
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
        for List in semi.itervalues():
            dist = [distances[li] for li in List]
            minIndex = dist.index(min(dist))
            (x_,y_) = List[minIndex]
    
            # dodanie do listy globalnej wierzchołków
            cornersInside.append((x_,y_))
        return cornersInside
    
    
    def findCorners(self):
        contours = self.contours
    
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
    
            indexes = self.findCornersOnContour(contours[cindex],16)
    
            # zaznacz wszsytkie znalezione wierzchołki
            for Id in indexes:
                (y,x) = contours[cindex][Id]
    
                # dodaj do globalnej listy ze wskazaniem konturu
                cornerCNT[cindex].append((x,y))
    
                # dodaj do listy bez wskazania konturu
                cornerList.append((x,y))
    
            # usunięcie podobnych punktów na konturze
            # cornerCNT[cindex] = eliminateSimilarCorners(cornerCNT[cindex],nimg,border=35)
    
        return cornerCNT,longestContour,cornerList
    
    
    def findObjects(self,contours):
        '''
         znajduje obiekty na podstawie konturów zmalezionych (łaczy poblisike kontury w jeden obiekt
        '''
    
        rectangles = []
        tmpbinary = np.zeros(self.shape,dtype='uint8')
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
        cntTMP, h = cv2.findContours(tmpbinary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
        return cntTMP
    
    
    def findMainObject(self,objectsCNT,img=0):
        '''
         znajduje obiekt najbardziej po srodku plaszczyzny (bo to nie beda krawedzie lustra)
        '''
        shape = self.shape
        #srodek obrazu
        yc0 = shape[0]/2
        xc0 = (shape[1]/4)*3
    
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
        mainCNT = objectsCNT[min_index]
    
        return mainCNT
    

