# -*- coding: utf-8 -*-
#/usr/bin/env python
'''
Created on Oct 31, 2014

@author: Tomasz
'''
import cv2
import numpy as np
from analyticGeometry import convertLineToGeneralForm
from zone import Zone

import func.markElements as mark

from calculations.labeling import LabelFactory
from ContourDectecting import ContourDetector


class objectDetector2(object):
    
    
        
    def __init__(self,md,image_origin):
        
        self.mirror_zone = md.mirrorZone
        self.mid = int(self.mirror_zone.offsetX+self.mirror_zone.width/2)
        self.md = md
        
        self.origin = image_origin
        
    def detect(self):
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
        
        k = 1
        kernel = np.ones((k,k))
        dilated = cv2.dilate(md.edges_mask,kernel)
        edge = np.where(dilated>0,255,0)
        
        zoneA =  Zone(edge,   mirror_zone.offsetX ,mirror_zone.offsetY                                ,mid-mirror_zone.offsetX                                    ,md.calculatePointOnLine(mid)[1]-mirror_zone.offsetY)
        zoneB =  Zone(edge,   mid                 ,mirror_zone.offsetY                                ,mirror_zone.offsetX+mirror_zone.width-mid                  ,md.calculatePointOnLine(mirror_zone.offsetX+mirror_zone.width)[1]-mirror_zone.offsetY)
        zoneC =  Zone(edge,   mirror_zone.offsetX ,md.calculatePointOnLine(mirror_zone.offsetX)[1]    ,mid-mirror_zone.offsetX                                    ,mirror_zone.offsetY+mirror_zone.height - md.calculatePointOnLine(mirror_zone.offsetX)[1])
        zoneD =  Zone(edge,   mid                 ,md.calculatePointOnLine(mid)[1]                    ,mirror_zone.offsetX+mirror_zone.width-mid                  ,mirror_zone.offsetY+mirror_zone.height - md.calculatePointOnLine(mid)[1] )
        
        margin = 50
        zoneA = self.setMargin(zoneA, margin)
        zoneB = self.setMargin(zoneB, margin)
        zoneC = self.setMargin(zoneC, margin)
        zoneD = self.setMargin(zoneD, margin)
        
        (x,y,w,h) = self.__findObject(zoneA.image)
        zoneA = Zone(self.origin,x+zoneA.offsetX,y+zoneA.offsetY,w,h)
        
        (x,y,w,h) = self.__findObject(zoneB.image)
        zoneB = Zone(self.origin,x+zoneB.offsetX,y+zoneB.offsetY,w,h)
        
        (x,y,w,h) = self.__findObject(zoneC.image)
        zoneC = Zone(self.origin,x+zoneC.offsetX,y+zoneC.offsetY,w,h)
        
        (x,y,w,h) = self.__findObject(zoneD.image)
        zoneD = Zone(self.origin,x+zoneD.offsetX,y+zoneD.offsetY,w,h)
        
        return zoneA,zoneB,zoneC,zoneD
    
    
    def setMargin(self,zone, margin):
        
        newOffsetX = zone.offsetX + margin 
        newOffsetY = zone.offsetY + margin
        newWidth = zone.width - margin - margin
        newHeight = zone.height - margin - margin
        
        zone = Zone(zone.origin, newOffsetX, newOffsetY, newWidth, newHeight)
        
        return zone 
        
        
    def __findObject(self,origin):
        lf = LabelFactory(origin)        
        lf.run(origin)
        lf.flattenLabels()
        
        contours = lf.convert2ContoursForm()
        
        cd = ContourDetector(origin)
        objects,margin = cd.findObjects(contours)
        
        center = []
        area = []
        rects = []
        small = []
        circles = []
        rectangles = []
        
        for j,BND in enumerate(objects):
            x,y,w,h = cv2.boundingRect(BND)
            
            #filtracja obiektow po prawej i lewej krawedzi oraz z dolu
            if x == 1 or x+w+1 == origin.shape[1] or y+h+1 == origin.shape[0]:
                
                center.append(None)
                area.append(None)
                rects.append(None)
                rectangles.append(None)
                
                continue
            
#             if y == 1 and letter in ['C','D']:
#                 continue
            center_i = (h/2,w/2)
            area_i = h*w
            
            center.append(center_i)
            area.append(area_i)
            rects.append((x,y,w,h))
            
            if abs(h-w) < max(h,w)*0.1:
                #przypadek kwadratu
                
                
                if abs ( margin*2 - w ) < w*0.1:
#                     print 'small square: ',(x,y, h, w)
                    small.append(j)
                    test1 = np.zeros_like(origin)
                    cv2.circle(test1,(int(x+w/2),int(y+h/2)),int(margin*2.5),200,1)
                    cv2.circle(origin,(int(x+w/2),int(y+h/2)),int(margin*2.5),200,1)
                    c1 = np.nonzero(test1)
                    c11 = np.transpose(c1)
                    circles.append(map(tuple,c11))
                    
                    
                #cv2.circle(origin,(int(x+w/2),int(y+h/2)),margin*2.5,200,2)
            test2 = np.zeros_like(origin)
            cv2.rectangle(test2,(x,y),(x+w,y+h),(255),1)
            cv2.rectangle(origin,(x,y),(x+w,y+h),(255),1)
            c2 = np.nonzero(test2)
            c22 = np.transpose(c2)
            rectangles.append(map(tuple,c22))
            
            
                
        iMax = area.index(max(area))
        
        common = []
        if len(circles) > 0:
            for c in circles:
#                 print 'common points'
                common_points = set(rectangles[iMax]).intersection(c)
                if len(common_points)>0:
#                     print 'YES'
                    common.append(c)
                else:
                    pass
#                     print 'NO'
        points = rectangles[iMax]            
        for c in common:
            points = points + c
        
        BND2 = np.asarray([[(y,x) for (x,y) in points]])
        x,y,w,h = cv2.boundingRect(BND2)
        cv2.rectangle(origin,(x,y),(x+w,y+h),(255),3) 
        
        padding = int ( w*0.1 )
        
        return (x ,y-padding,w,h)
   