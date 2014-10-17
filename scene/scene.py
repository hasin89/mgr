# -*- coding: utf-8 -*-
'''
Created on Sep 3, 2014

@author: Tomasz
'''
import func.trackFeatures as features
import cv2
from edge import edgeMap
from md5 import blocksize
import numpy as np


class Scene(object):
    u'''
    Reprezentuje cały obraz sceny
    '''
    
    
    def __init__(self, image):
        self.view = image
        self.height, self.width = self.view.shape[:2]
        
        self.reflected = None
        self.direct = None
        
        # edge detection parameters
        self.gauss_kernel = None
        self.gamma = None
        
        self.constant = 0
        self.blockSize = 0
        
        self.edge_map = None
        
        self.getGrayScaleImage()
        
        
        
    def getEdges(self):
        
        #edge_filtred, vis = features.canny(self.view, self.gauss_kernel, self.gamma)
        edge_filtred, vis = features.adaptiveThreshold(self.view, self.gauss_kernel, self.gamma,self.constant,self.blockSize)
        self.edge_map = edgeMap(self.view, edge_filtred)
        
        return self.edge_map
    
    def getEdges2(self,gauss_kernel=5,constant=5,blockSize=101,treshold=4):
        
#         gauss_kernel = 5
#         constant = 5
#         blockSize = 101
#         tresh = 4
        
        gray = self.gray
        gray_filtred = cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)
        
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtred,
                                             maxValue=255,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=blockSize,
                                             C=constant)
        
        dst = cv2.distanceTransform(edge_filtred,cv2.cv.CV_DIST_C,3)
        mask = np.where(dst>treshold,255,0).astype('uint8')
        self.edge_map = edgeMap(self.view, mask)
        return self.edge_map
    
    def getGrayScaleImage(self):
        self.gray = cv2.cvtColor(self.view, cv2.COLOR_BGR2GRAY)
        return self.gray
    
    def divide(self, mirror_line=None):
        if mirror_line == None:
            mirror_line = self.mirror_line
        
#         Ax+By+c self.width
        y = abs((mirror_line[0]*self.width/2.0+mirror_line[2])/mirror_line[1])
        reflected = Scene(self.view[:y, :])
        direct = Scene(self.view[y:, :])
        
        return reflected, direct
