# -*- coding: utf-8 -*-
'''
Created on Sep 3, 2014

@author: Tomasz
'''
import cv2
import numpy as np
from func import histogram
import func.trackFeatures as features
import Draw
from CornerDectecting import CornerDetector

class spottedObject(object):
    u'''
        reprezentuje wykryty obiekt 
    '''


    def __init__(self,CNT):
        '''
        Constructor
        '''
        self.CNT = CNT
        x,y,w,h = cv2.boundingRect(self.CNT)
        
        # SqrBnd zawiera  wielokatna obwiednie bryły ? a nie prostokątną?
        self.sqrBnd = (x,y,w,h)
        self.contours = None
        self.corners = None

    
    def markOnCanvas(self,canvas,color):
        """
        
        """
        for i in range(0,len(self.CNT)-1):
            Draw.Segment(canvas,(self.CNT[i][0][0],self.CNT[i][0][1]) ,(self.CNT[i+1][0][0],self.CNT[i+1][0][1]),color)
            print str((self.CNT[i][0][0],self.CNT[i][0][1]))
            
    def setContours(self,contours):
        """
        nastepca funkcji filterContours
        z podanych konturów zostawia tylko te związane z obiektem
        
        pozbywa sie konturów z poza podanego obszaru
    
        contours - kontury - {0:[(a,b),(c,d)],1:[(e,f),(g,h),(j,k)]}
        boundaries - obszar graniczy - [[[1698  345]] \n\n [[1698  972]]]
    
        """
        toDel = []
        boundaries = self.CNT
        for key,c in contours.iteritems():
            if len(c)>0:
                isinside = cv2.pointPolygonTest(boundaries,(c[0][1],c[0][0]),0)
            else:
                isinside = 0
            if isinside != 1:
                contours[key] = []
            else:
                pass
        self.contours = contours
        return self.contours
    
    
    def getCorners(self):
        cd = CornerDetector(self.contours)
        corners = cd.findCorners()
        self.corners = corners
        pass
        
        