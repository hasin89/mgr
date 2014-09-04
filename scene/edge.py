# -*- coding: utf-8 -*-
'''
Created on Sep 3, 2014

@author: Tomasz
'''
import cv2
import numpy as np
from func import histogram
import func.trackFeatures as features

class edgeMap(object):
    u'''
        reprezentuje binarną mapę wykrytych konturów 
    '''


    def __init__(self,origin,binaryMap):
        '''
        Constructor
        '''
        self.origin = origin
        self.map = binaryMap
        
        self.height = binaryMap.shape[0]
        self.width = binaryMap.shape[1]
        self.shape = (self.height,self.width)
        
        self.mirror_line = None
        
        self.reflected = None
        self.direct = None
        
        self.contours = None
        self.objects = None
        
        
        
    def countNonZeroRowsX(self):
        peri= []
        for i in range(self.width):
            peri.append(len(np.nonzero(self.map[:,i])[0]))
        non = np.nonzero(peri)
        c,d = non[0][0],non[0][-1]
        h = histogram.draw(peri[c:d])
        return h,c,d
        
        
    def getMirrorLine(self):
        
        peri= []
        for i in range(self.height):
            p = len(np.nonzero(self.map[i])[0])
            peri.append(p)
        #pierwsze kontury
        non = np.nonzero(peri)
        a,b = non[0][0],non[0][-1]
    
        h,a,b = histogram.draw2(peri)
        
        self.mirror_line = [a,b]
        # return h,non[0][0],non[0][-1],[a,b]
        return self.mirror_line
    
    def divide(self,mirror_line=None):
        if mirror_line == None:
            mirror_line = self.mirror_line
        self.reflected = self.map[:mirror_line[0],:]
        self.direct = self.map[mirror_line[1]:,:]
        
    def findObject(self,img=0):
        '''
        znajduje kontury, rogi konturu,
        return (mainBND,mainSqrBnd,contours,objects)
        '''
    
        self.contours = features.findContours(self.map)
    
        # objects zawiera obiekty znalezione na podstawie konturu
        self.objects = features.findObjects(self.shape,self.contours)
    
        for tmpobj in self.objects:
            for i in range(0,len(tmpobj)-1):
                # mark.drawSegment(img,(tmpobj[i][0][0],tmpobj[i][0][1]) ,(tmpobj[i+1][0][0],tmpobj[i+1][0][1]))
                pass
    
    
        # obiekt główny
        mainBND = features.findMainObject(objects,shape,img)
        for i in range(0,len(mainBND)-1):
                mark.drawMain(img,(mainBND[i][0][0],mainBND[i][0][1]) ,(mainBND[i+1][0][0],mainBND[i+1][0][1]))
    
        x,y,w,h = cv2.boundingRect(mainBND)
        # mainSqrBnd zawiera  wielokatna obwiednie bryły
        mainSqrBnd = (x,y,w,h)
    
        return mainBND,mainSqrBnd,contours,objects
    
        
