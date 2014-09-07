# -*- coding: utf-8 -*-
from math import sqrt
from spottedObject import spottedObject

__author__ = 'tomek'
#/usr/bin/env python

'''

'''

import cv2
import numpy as np


class ObjectDetector():
    
    
    def __init__(self,contours,shape):
        self.contours = contours
        self.shape = shape
        
    
    def findObjects(self):
        '''
         znajduje obiekty na podstawie konturów zmalezionych (łaczy poblisike kontury w jeden obiekt
        '''
        contours = self.contours
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
    
    
    def findMainObject(self,objectsCNT):
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
        
        mainObject = spottedObject(mainCNT)
        mainObject.setContours(self.contours)
    
        return mainObject
    

