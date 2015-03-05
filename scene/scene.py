# -*- coding: utf-8 -*-
'''
Created on Sep 3, 2014

@author: Tomasz
'''
import func.trackFeatures as features
import cv2
import numpy as np


class Scene(object):
    u'''
    Reprezentuje cały obraz sceny
    '''
    gray = None
    
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
