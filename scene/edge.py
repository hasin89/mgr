# -*- coding: utf-8 -*-
'''
Created on Sep 3, 2014

@author: Tomasz
'''
import cv2
import numpy as np
from func import histogram
from ContourDectecting import ContourDetector
from ObjectDectecting import ObjectDetector
from analyticGeometry import convertLineToGeneralForm


class edgeMap(object):
    u'''
        reprezentuje binarną mapę wykrytych konturów 
    '''


    def __init__(self, origin, binaryMap):
        '''
        Constructor
        '''
        self.view = origin
        self.map = binaryMap
        
        self.height = binaryMap.shape[0]
        self.width = binaryMap.shape[1]
        self.shape = (self.height, self.width)
        
        self.mirror_line = None
        
        self.contours = None
        self.objects = None
        self.mainObject = None
        
        
        
    def countNonZeroRowsX(self):
        peri = []
        for i in range(self.width):
            peri.append(len(np.nonzero(self.map[:, i])[0]))
        non = np.nonzero(peri)
        c, d = non[0][0], non[0][-1]
        h = histogram.draw(peri[c:d])
        return h, c, d
        
        
    def getMirrorLine(self):
        """
        znajduje linie styku lusrta z podłożem
        """
        
        peri = []
        for i in range(self.height):
            p = len(np.nonzero(self.map[i])[0])
            peri.append(p)
        # pierwsze kontury
        non = np.nonzero(peri)
        a, b = non[0][0], non[0][-1]
    
        h, a, b = histogram.draw2(peri)
        
        self.mirror_line = [a, b]
        # return h,non[0][0],non[0][-1],[a,b]
        return self.mirror_line
    
    def getMirrorLine2(self):
        rho = 1.5
        # theta = 0.025
        theta = np.pi/180
        
        threshold=int(self.map.shape[1]/3)
        lines2 = cv2.HoughLines(self.map,rho,theta,threshold)
        
        Amin = 2
        for (rho,theta) in lines2[0][:2]:
            print (rho,theta)
            line = convertLineToGeneralForm((rho,theta),self.map.shape)
            A = abs((round(line[0],0)))
            if A<Amin:
                self.mirror_line = line
                Amin=A 
        
        return self.mirror_line
   
    
    def divide(self, mirror_line=None):
        if mirror_line == None:
            mirror_line = self.mirror_line
        
#         Ax+By+c self.width
        y = abs((mirror_line[0]*self.width/2.0+mirror_line[2])/mirror_line[1])
        reflected = edgeMap(self.view,self.map[:y, :])
        direct = edgeMap(self.view,self.map[y:, :])
        
        return reflected, direct
    
    def findObject(self):
        '''
        znajduje kontury, rogi konturu,
        return (mainBND,mainSqrBnd,contours,objects)
        '''
    
        shape = (self.map.shape[0],self.map.shape[1])
        
        cd = ContourDetector(self.map)
    
        self.contours = cd.findContours()
        
        od = ObjectDetector(self.contours,self.shape)
    
        # objects zawiera obiekty znalezione na podstawie konturu
        self.objects = od.findObjects()
    
        # obiekt główny
        self.mainObject = od.findMainObject(self.objects)
        
    
        return self.mainObject,self.contours,self.objects
    
        
