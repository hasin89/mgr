# -*- coding: utf-8 -*-
'''
Created on Oct 31, 2014

@author: Tomasz
'''
import numpy as np
import cv2

class Zone(object):
    
    offsetX = None
    
    offsetY = None
    
    def __init__(self,image,x,y,width,height,mask=None):
        self.offsetX = x
        self.offsetY = y
        
        self.width = width
        self.height = height

#         self.scene.view[y:, :]
        
        #smaller image for calculations
        self.origin = image
        
        if mask is not None:
            self.mask = mask
        else:
            mask = np.zeros(image.shape[:2],dtype='uint8')
            roi = np.ones((height,width),dtype='uint8')
            mask[y:y+height, x:x+width] = roi
            self.mask = mask
        
        #local and global image with mask
        self.getFiltredImge()    
                
        
    def getPreview(self):
        m3 = self.mask.reshape(self.mask.shape[0],self.mask.shape[1],1)
        self.preview = self.origin*m3
        
        return self.preview
    
    def getFiltredImge(self):
        m3 = self.mask.reshape(self.mask.shape[0],self.mask.shape[1],1)
        self.getPreview()
        self.image = self.preview[self.offsetY:self.offsetY+self.height, self.offsetX:self.offsetX+self.width]
        
        return self.image
    
    def clearMask(self):
        self.mask = np.ones(self.origin.shape[:2],dtype='uint8')