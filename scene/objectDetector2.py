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
            
#         margin = 20
#         zoneA = self.setMargin(zoneA, margin)
#         zoneC = self.setMargin(zoneC, margin)
        
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
        
#         labels =  np.unique(labelsMap)
#         for label in labels:
#             if label == -1:
#                 continue
#             objects = np.where(labelsMap == label,1,0)
#             indices = np.nonzero(objects)
#             print indices
#             points =  np.array([ indices[0],indices[1] ]).T 
#             print points
#             y,x,h,w = cv2.boundingRect(objects)
#             print y,x,h,w
        
#         cd = ContourDetector(origin)
#         objects,margin = cd.findObjects(contours)
#         
#         center = []
#         area = []
#         rects = []
#         small = []
#         circles = []
#         rectangles = []
#         
#         for j,BND in enumerate(objects):
#             x,y,w,h = cv2.boundingRect(BND)
#             
#             #filtracja obiektow po prawej i lewej krawedzi oraz z dolu
#             if x == 1 or x+w+1 == origin.shape[1] or y+h+1 == origin.shape[0]:
#                 
#                 center.append(None)
#                 area.append(None)
#                 rects.append(None)
#                 rectangles.append(None)
#                 
#                 continue
#             
# #             if y == 1 and letter in ['C','D']:
# #                 continue
#             center_i = (h/2,w/2)
#             area_i = h*w
#             
#             center.append(center_i)
#             area.append(area_i)
#             rects.append((x,y,w,h))
#             
#             if abs(h-w) < max(h,w)*0.1:
#                 #przypadek kwadratu
#                 
#                 
#                 if abs ( margin*2 - w ) < w*0.1:
# #                     print 'small square: ',(x,y, h, w)
#                     small.append(j)
#                     test1 = np.zeros_like(origin)
#                     cv2.circle(test1,(int(x+w/2),int(y+h/2)),int(margin*2.5),200,1)
#                     cv2.circle(origin,(int(x+w/2),int(y+h/2)),int(margin*2.5),200,1)
#                     c1 = np.nonzero(test1)
#                     c11 = np.transpose(c1)
#                     circles.append(map(tuple,c11))
#                     
#                     
#                 #cv2.circle(origin,(int(x+w/2),int(y+h/2)),margin*2.5,200,2)
#             test2 = np.zeros_like(origin)
#             cv2.rectangle(test2,(x,y),(x+w,y+h),(255),1)
#             cv2.rectangle(origin,(x,y),(x+w,y+h),(255),1)
#             c2 = np.nonzero(test2)
#             c22 = np.transpose(c2)
#             rectangles.append(map(tuple,c22))
#             
# #       mark object DEBUG
# #         for i in range(0,len(BND)-1):
# #             cv2.line(origin,(BND[i][0][0],BND[i][0][1]) ,(BND[i+1][0][0],BND[i+1][0][1]),255,1)
#                     
#         iMax = area.index(max(area))
#         print iMax
#         common = []
#         if len(circles) > 0:
#             for c in circles:
# #                 print 'common points'
#                 common_points = set(rectangles[iMax]).intersection(c)
#                 if len(common_points)>0:
# #                     print 'YES'
#                     common.append(c)
#                 else:
#                     pass
# #                     print 'NO'
#         points = rectangles[iMax]            
#         for c in common:
#             points = points + c
#         
#         BND2 = np.asarray([[(y,x) for (x,y) in points]])
#         x,y,w,h = cv2.boundingRect(BND2)
#         cv2.rectangle(origin,(x,y),(x+w,y+h),(255),3) 
#         
#         padding = int ( w*0.1 )
#         
#         Y = y-padding
#         
#         # ewentualne dodawanie oderwanych czesci duzych kwadratow
#         maxi = rects[iMax]
#         
#         if len(area)>1:
#             area[iMax] = None
#             iMax2 = area.index(max(area))
#             
#             if area[iMax2] != None:
#         
#                 midi = rects[iMax2]
#                 
#                 if midi[0]>maxi[0] and midi[1]<maxi[1] and midi[2]<maxi[2]:
#                     Y = midi[1]
#                     h += midi[3] + abs(maxi[1] - midi[1]-midi[3] )
                    
        return (x ,Y,w,h)
   
   