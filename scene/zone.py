# -*- coding: utf-8 -*-
'''
Created on Oct 31, 2014

@author: Tomasz
'''
import numpy as np

class Zone(object):
    
    offsetX = None
    
    offsetY = None
    
    def __init__(self,image,x,y,width,height,mask=None):
        self.offsetX = x
        self.offsetY = y
        
        self.width = width
        self.height = height
        
        self.image = None
        self.preview = None

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
        '''
        return ROI on the original image canvas
        determined by mask 
        the same size as original
        global image
        '''
#         print 'origin shape' + str(self.origin.shape)
        if len(self.origin.shape) == 3:
            m3 = self.mask.reshape(self.mask.shape[0],self.mask.shape[1],1)
        else:
            m3 = self.mask
        self.preview = self.origin*m3
        
        return self.preview
    
    
    def getFiltredImge(self):
        '''
        return image of ROI. This image is fragment of the original one and it is smaller.
        Determined by mask
        local image
        '''
        self.getPreview()
        self.image = self.preview[self.offsetY:self.offsetY+self.height, self.offsetX:self.offsetX+self.width]
        
        return self.image
    
    
    def clearMask(self):
        '''
        clear the mask
        '''
        self.mask = np.ones(self.origin.shape[:2],dtype='uint8')
        
        
    def setMargin(self,margin):
        
        newOffsetX = self.offsetX + margin 
        newOffsetY = self.offsetY + margin
        newWidth = self.width - margin - margin
        newHeight = self.height - margin - margin
        
        s = self.__init__(self.origin, newOffsetX, newOffsetY, newWidth, newHeight)
        
        return s 
    
    def getRectangle(self):
        return self.offsetX,self.offsetY,self.width,self.height