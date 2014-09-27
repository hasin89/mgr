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
    
    
    def __init__(self,shape,contours):
        self.contours = contours
        self.shape = shape
     
    #
    # podstawowa analiza obrazu
    #  
    
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
    
      
    def eliminateSimilarCorners(self,corners,mainCnt,border=35):
        '''
        eliminuje wierzchołki ktore prawdopodobnie sa blisko siebie
        '''
        shape = self.shape
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
    
        #dla każdego znalezionego konturu
        for cindex in range(len(contours)):
            cornerCNT[cindex] = []
         
            indexes = self.findCornersOnContour(contours[cindex],16)
    
            # zaznacz wszsytkie znalezione wierzchołki
            for Id in indexes:
                (y,x) = contours[cindex][Id]
    
                # dodaj do globalnej listy ze wskazaniem konturu
                cornerCNT[cindex].append((x,y))
    
                # dodaj do listy bez wskazania konturu
                cornerList.append((x,y))
    
            # usunięcie podobnych punktów na konturze TODO: whtf ? czemu zakomentowane? bo jest późnej wywołane dla wszystkich zbiorczo w kontekście głównego konturu
            # cornerCNT[cindex] = eliminateSimilarCorners(cornerCNT[cindex],nimg,border=35)
    
        #return cornerCNT,longestContour,cornerList
        return cornerList
    
    
    #
