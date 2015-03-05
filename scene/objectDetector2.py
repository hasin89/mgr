# -*- coding: utf-8 -*-
#/usr/bin/env python
'''
Created on Oct 31, 2014

@author: Tomasz
'''
import cv2
import numpy as np
from zone import Zone


from calculations.labeling import LabelFactory


class objectDetector2(object):
    
    mirrorOfsets = None
        
    def __init__(self,md,image_origin,mirrorOffsets=None):
        
        self.mirrorOffsets = mirrorOffsets
        
        self.mirror_zone = md.mirrorZone
        self.mid = int(self.mirror_zone.offsetX+self.mirror_zone.width/2)
        self.md = md
        
        self.origin = image_origin
        
    def detect2(self,md):
        
        img  = md.origin
        k = 1
        kernel = np.ones((k,k))
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        #edges mask
        gauss_kernel = 5
        tresholdMaxValue = 1
        blockSize = 15
        constant = -2
        
        distanceTreshold = 2
        
        gray_filtred = cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtred,
                                             maxValue=tresholdMaxValue,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                             thresholdType=cv2.THRESH_BINARY,
                                             blockSize=blockSize,
                                             C=constant)
        
        
        # CV_DIST -> sasiedztwo 8 spojne
        dst = cv2.distanceTransform(edge_filtred,cv2.cv.CV_DIST_C,3) # 3 to jest wielkość maski, która nie ma znaczenia dla 8 spojnego sasiedztwa
        
        
#         
#         # znajdz punkty o odleglosci wiekszej niz prog. generalnie grube krechy
        mask = np.where(dst>distanceTreshold,1,0).astype('uint8')
        
        binary = mask
        
        mirror_zone = md.mirrorZone
        
        zoneMask = md.filterDown
        
        zc = np.where(zoneMask == 1,binary,0).astype('uint8')
        za = np.where(zoneMask == 1,0,binary).astype('uint8')
        
        
        zoneA =  Zone(za,   0 ,0 , mirror_zone.width ,mirror_zone.height, np.where(zoneMask == 0,1,0))
        zoneC =  Zone(zc,   0 ,0 , mirror_zone.width, mirror_zone.height, zoneMask)
        
        if self.mirrorOffsets is not None:
            os = self.mirrorOffsets[0][1]
            scan = zoneA.image.copy()
            scan[:,os:] = 0
            zoneA.image = scan.copy()
            
            final = np.zeros_like(img)
            final[scan==1] = (255,255,255)
#         cv2.imwrite('results/test.jpg', final)
        
            os = self.mirrorOffsets[1][1]
            zoneC.image[:,os:] = 0
        
            final = np.zeros_like(img)
            scan = zoneC.image.copy()
            scan[:,os:] = 0
            zoneC.image = scan.copy()
        
            final[scan==1] = (255,255,255)
            cv2.imwrite('results/test.jpg', final)
        
        
        return zoneA, zoneC
        
        
    def detect(self,chessboard=False,multi=False):
        '''
        detects objects
        
         A|B
         ---
         C|D
        
        on the image, basing on detected mirror zone
        '''
        
        mirror_zone = self.mirror_zone
        mid = self.mid
        md = self.md
        
        edges_mask = ''
        img = mirror_zone.image
        
        img = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs = 180
        thrs = chesboardTreshold = 180
        retval, binar = cv2.threshold(gray, thrs, 1, cv2.THRESH_BINARY)
        size = 101
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        binar = cv2.dilate(binar,cross)
        binar = cv2.erode(binar,cross)
        
#         bb = np.zeros_like(img)
#         bb= np.where(binar == 1,255,0)
#         
#         fi = '100.jpg'
#         cv2.imwrite('results/objects_binary_%s'% fi,bb)
        
        #detection
        
        k = 4
        kernel = np.ones((k,k))
        dilated = cv2.dilate(md.edges_mask,kernel)
        edge = np.where(dilated>0,255,0)
        
        zoneA, zoneC = self.detect2(md)
            
        print 'results/zoneA.jpg'
        cv2.imwrite('results/zoneA.jpg',np.where(zoneA.image == 1,255,0))
        cv2.imwrite('results/zoneC.jpg',np.where(zoneC.image == 1,255,0))
                
        (x,y,w,h) =  self.__findObject(zoneA.image,0)
        zoneA = Zone(self.origin,x+zoneA.offsetX,y+zoneA.offsetY,w,h)
        
        (x,y,w,h) = self.__findObject(zoneC.image,1)
        zoneC = Zone(self.origin,x+zoneC.offsetX,y+zoneC.offsetY,w,h)
        
        margin = -20
        zoneA = self.setMargin(zoneA, margin)
        zoneC = self.setMargin(zoneC, margin)
        
        cv2.imwrite('results/finalZoneA.jpg',zoneA.image)
        cv2.imwrite('results/finalZoneC.jpg',zoneC.image)
        
        return zoneA,zoneC
    
    def __getROI(self,image_type):
        if image_type == 0:
            mask = np.zeros(self.origin.shape[:2])
            p1 = (1779,978)
            p2 = (2353,1433)
            p3 = (197,2297)
            p4 = (165,1261)
            triangle = np.array([ p1, p2, p3 ,  p4], np.int32)
            cv2.fillConvexPoly(mask, triangle, 1)
            
        if image_type == 1:
            mask = np.zeros(self.origin.shape[:2])
            p1 = (2581,1497)
            p2 = (3589,2005)
            p3 = (2705,3053)
            p4 = (25,2713)
            triangle = np.array([ p1, p2, p3 ,  p4], np.int32)
            cv2.fillConvexPoly(mask, triangle, 1)
        
        return mask
            
    
    def setMargin(self,zone, margin):
        
        newOffsetX = zone.offsetX + margin 
        newOffsetY = zone.offsetY + margin
        newWidth = zone.width - margin - margin
        newHeight = zone.height - margin - margin
        
        zone = Zone(zone.origin, newOffsetX, newOffsetY, newWidth, newHeight)
        
        return zone 
        
        
    def __findObject(self,origin,image_type):
        
        binary = origin.copy()
               
        shape = origin.shape[:2]
        
        mask = self.__getROI(image_type)
        binary = np.where(mask==1,binary,0).astype('uint8')
               
        size = 3
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(size,size))
        binary = cv2.erode(binary,cross)    
        binary = cv2.dilate(binary,cross)
        
        size = 50
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        binary = cv2.dilate(binary,cross)
        
        
        cv2.imwrite('results/labeling.jpg',np.where(binary==1,255,0))  
                 
        lf = LabelFactory([])   
        
        labelsMap = lf.getLabelsExternal(binary, 8, 0)
        
        Max = 0
        for l in np.unique(labelsMap):
            if l == -1:
                continue
            Map = np.where(labelsMap == l,255,0)
            count = np.count_nonzero(Map)
            
            if count > Max :
                Max = count
                maxLabel = l 
        print maxLabel, Max
        Map = np.where(labelsMap == maxLabel,255,0)
        
        points = np.nonzero(Map)
        yy = points[0]
        xx = points[1]
        print max(xx), min(xx), max(yy), min(yy)
        x = min(xx)
        Y = min(yy)
        w = max(xx) - min(xx)
        h = max(yy) - min(yy)
        
        Map[:,x] = 1
        Map[:,max(xx)] = 1
        Map[max(yy),:] = 1
        Map[Y,:] = 1
        cv2.imwrite('results/labeling1_%d_.jpg'%l,Map)
                    
        return (x ,Y,w,h)
   
   