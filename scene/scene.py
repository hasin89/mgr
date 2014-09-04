# -*- coding: utf-8 -*-
'''
Created on Sep 3, 2014

@author: Tomasz
'''
import func.trackFeatures as features
import cv2
from edge import edgeMap


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
        
        self.edge_map = None
        
        self.getGrayScaleImage()
        
        
        
    def getEdges(self):
        
        edge_filtred, vis = features.canny(self.view, self.gauss_kernel, self.gamma)
        self.edge_map = edgeMap(self.view, edge_filtred)
        
        return self.edge_map
    
    def getGrayScaleImage(self):
        self.gray = cv2.cvtColor(self.view, cv2.COLOR_BGR2GRAY)
        return self.gray
    
    def divide(self, mirror_line=None):
        if mirror_line == None:
            mirror_line = self.mirror_line
        reflected = Scene(self.view[:mirror_line[0], :])
        direct = Scene(self.view[mirror_line[1]:, :])
        
        return reflected, direct
