# -*- coding: utf-8 -*-
'''
Created on Oct 31, 2014

@author: Tomasz
'''
import numpy as np

class Zone(object):
    
    offsetX = None
    
    offsetY = None
    
    def __init__(self,image,x,y,width,height):
        self.offsetX = x
        self.offsetY = y
        
#         self.scene.view[y:, :]
        
        self.image = image[y:y+height, x:x+width]
        mask = np.zeros(image.shape[:2],dtype='uint8')
        
        roi = np.ones((height,width),dtype='uint8')
        
        self.mask = mask
        self.mask[y:y+height, x:x+width] = roi
        
        self.preview = image.copy()
        self.preview[:,:,:] = (0,0,0)
        self.preview[y:y+height, x:x+width] = self.image
                
        self.width = width
        self.height = height