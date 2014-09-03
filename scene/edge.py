# -*- coding: utf-8 -*-
'''
Created on Sep 3, 2014

@author: Tomasz
'''
import cv2
import numpy as np
from func import histogram

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
        
        self.mirror_line = None
        
        self.reflected = None
        self.direct = None
        
        
        
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
    
        
