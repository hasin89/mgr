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
        self.image = image
        self.height, self.width = self.image.shape[:2]
        
        self.reflected = None
        self.direct = None
        
        # edge detection parameters
        self.gauss_kernel = None
        self.gamma = None
        
        self.edge_map = None
        
        self.getGrayScaleImage()
        
        
        
    def getEdges(self):
        
        edge_filtred, vis = features.canny(self.image, self.gauss_kernel, self.gamma)
        self.edge_map = edgeMap(self.image, edge_filtred)
        
        return self.edge_map
    
    def getGrayScaleImage(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.gray
    
    def divide(self, mirror_line=None):
        if mirror_line == None:
            mirror_line = self.mirror_line
        self.reflected = Scene(self.image[:mirror_line[0], :])
        self.direct = Scene(self.image[mirror_line[1]:, :])
