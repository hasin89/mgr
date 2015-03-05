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
#         
#         edges_mask = mask
#         
# #         edges_mask = md.edges_mask
#         
#         dilated = cv2.dilate(mask,kernel)
#         mask = np.where(dilated>0,1,0)
                
        final = np.zeros_like(img)
        final[mask==1] = (255,255,255)
        
        binary = mask
        
        cv2.imwrite('results/test.jpg', final)
        
        mid = int(self.mirror_zone.offsetX+self.mirror_zone.width/2)
        y = md.calculatePointOnLine(mid)[1]
        mirror_zone = md.mirrorZone
        zoneA =  Zone(binary,   0 ,0 , mirror_zone.width ,y)
        zoneC =  Zone(binary,   0 ,y , mirror_zone.width, mirror_zone.height - y)
        
        os = self.mirrorOffsets[0][1]
        
        final = np.zeros_like(img)
        scan = zoneA.image.copy()
        scan[:,os:] = 0
        zoneA.image = scan.copy()
        
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
        
        bb = np.zeros_like(img)
        bb= np.where(binar == 1,255,0)
        
        fi = '100.jpg'
        cv2.imwrite('results/objects_binary_%s'% fi,bb)
        
        #detection
        
        k = 4
        kernel = np.ones((k,k))
        dilated = cv2.dilate(md.edges_mask,kernel)
        edge = np.where(dilated>0,255,0)
        
        if False:
            if chessboard:
                if multi == True:
                    y = md.calculatePointOnLine(mirror_zone.offsetX)[1]
                    zoneA =  Zone(edge,   0 ,0 ,mirror_zone.width ,y)
                    zoneC =  Zone(edge,   mirror_zone.offsetX ,y , mirror_zone.width,mirror_zone.height - y)
                else:
                    zoneA =  Zone(edge,   mirror_zone.offsetX ,mirror_zone.offsetY                                ,mid-mirror_zone.offsetX                                    ,md.calculatePointOnLine(mid)[1]-mirror_zone.offsetY)
                    zoneC =  Zone(edge,   mirror_zone.offsetX ,md.calculatePointOnLine(mirror_zone.offsetX)[1]    ,mid-mirror_zone.offsetX                                    ,mirror_zone.offsetY+mirror_zone.height - md.calculatePointOnLine(mirror_zone.offsetX)[1])
                    
            if not chessboard:
                zoneA =  Zone(edge,   mirror_zone.offsetX ,mirror_zone.offsetY                                ,mid-mirror_zone.offsetX                                    ,md.calculatePointOnLine(mid)[1]-mirror_zone.offsetY)
                zoneC =  Zone(edge,   mirror_zone.offsetX ,md.calculatePointOnLine(mirror_zone.offsetX)[1]    ,mid-mirror_zone.offsetX                                    ,mirror_zone.offsetY+mirror_zone.height - md.calculatePointOnLine(mirror_zone.offsetX)[1])
                zoneB =  Zone(edge,   mid                 ,mirror_zone.offsetY                                ,mirror_zone.offsetX+mirror_zone.width-mid                  ,md.calculatePointOnLine(mirror_zone.offsetX+mirror_zone.width)[1]-mirror_zone.offsetY)
                zoneD =  Zone(edge,   mid                 ,md.calculatePointOnLine(mid)[1]                    ,mirror_zone.offsetX+mirror_zone.width-mid                  ,mirror_zone.offsetY+mirror_zone.height - md.calculatePointOnLine(mid)[1] )
        else:
            
            zoneA, zoneC = self.detect2(md)
            
#         margin = 20
#         zoneA = self.setMargin(zoneA, margin)
#         zoneC = self.setMargin(zoneC, margin)
        
        print 'results/zoneA.jpg'
        cv2.imwrite('results/zoneA.jpg',np.where(zoneA.image == 1,255,0))
        cv2.imwrite('results/zoneC.jpg',np.where(zoneC.image == 1,255,0))
                
        (x,y,w,h) = self.__findObject(zoneA.image)
        zoneA = Zone(self.origin,x+zoneA.offsetX,y+zoneA.offsetY,w,h)
        
        (x,y,w,h) = self.__findObject(zoneC.image)
        zoneC = Zone(self.origin,x+zoneC.offsetX,y+zoneC.offsetY,w,h)
        
        cv2.imwrite('results/zoneA.jpg',zoneA.image)
        cv2.imwrite('results/zoneC.jpg',zoneC.image)
        
        return zoneA,zoneC
        
        if not chessboard:
            (x,y,w,h) = self.__findObject(zoneB.image)
            zoneB = Zone(self.origin,x+zoneB.offsetX,y+zoneB.offsetY,w,h)
            
            (x,y,w,h) = self.__findObject(zoneD.image)
            zoneD = Zone(self.origin,x+zoneD.offsetX,y+zoneD.offsetY,w,h)
        
        if not chessboard:
            return zoneA,zoneB,zoneC,zoneD
        else:
            return zoneA,zoneC
    
    
    def setMargin(self,zone, margin):
        
        newOffsetX = zone.offsetX + margin 
        newOffsetY = zone.offsetY + margin
        newWidth = zone.width - margin - margin
        newHeight = zone.height - margin - margin
        
        zone = Zone(zone.origin, newOffsetX, newOffsetY, newWidth, newHeight)
        
        return zone 
        
        
    def __findObject(self,origin):
        
        binary = origin.copy()
               
        shape = origin.shape[:2]
               
        size = 20
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        binary = cv2.dilate(binary,cross)
#         binary = cv2.dilate(binary,cross)
#         binary = cv2.dilate(binary,cross)

        rho = 1
        theta = np.pi/90
        threshold = shape[0]/2
        
        lines = cv2.HoughLines(binary, rho, theta, threshold)
        
        img = np.zeros((shape[0],shape[1],3))
        
        img = np.where(binary==1,255,0)
            
        if lines is not None:
            m,n = img.shape
            for (rho, theta) in lines[0]:
            # blue for infinite lines (only draw the 5 strongest)
                x0 = np.cos(theta)*rho
                y0 = np.sin(theta)*rho
                pt1 = ( int(x0 + (m+n)*(-np.sin(theta))), int(y0 + (m+n)*np.cos(theta)) )
                pt2 = ( int(x0 - (m+n)*(-np.sin(theta))), int(y0 - (m+n)*np.cos(theta)) )
                
                c = np.cos( abs(theta - np.pi/2) )
                if c >0.8:
                    mark.drawHoughLines([[rho, theta]], img, (128,0,128), 5)
                    margin = 20
                    triangle = np.array([ (pt1[0],pt1[1]-margin), (pt2[0],pt2[1]-margin), (shape[0],shape[1]) ], np.int32)
                    cv2.fillConvexPoly(binary, triangle, 0)
                    
                if c <0.4:
                    mark.drawHoughLines([[rho, theta]], img, (128,0,128), 5)
                    triangle = np.array([ (pt1[0]+margin,pt1[1]), (pt2[0]+margin,pt2[1]), (shape[0],0), (0,0) ], np.int32)
                    cv2.fillConvexPoly(binary, triangle, 0)
        
        size = 50
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        binary = cv2.dilate(binary,cross)
        binary = cv2.erode(binary,cross)        
            
        cv2.imwrite('results/lines.jpg',img)    
        cv2.imwrite('results/labeling.jpg',np.where(binary==1,255,0))
                 
        lf = LabelFactory(binary)   
        cv2.imwrite('results/labeling.jpg',np.where(binary==1,255,0))     
        lf.run(binary)
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
            
#       mark object DEBUG
#         for i in range(0,len(BND)-1):
#             cv2.line(origin,(BND[i][0][0],BND[i][0][1]) ,(BND[i+1][0][0],BND[i+1][0][1]),255,1)
                    
        iMax = area.index(max(area))
        print iMax
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
        
        Y = y-padding
        
        # ewentualne dodawanie oderwanych czesci duzych kwadratow
        maxi = rects[iMax]
        
        if len(area)>1:
            area[iMax] = None
            iMax2 = area.index(max(area))
            
            if area[iMax2] != None:
        
                midi = rects[iMax2]
                
                if midi[0]>maxi[0] and midi[1]<maxi[1] and midi[2]<maxi[2]:
                    Y = midi[1]
                    h += midi[3] + abs(maxi[1] - midi[1]-midi[3] )
                    
        return (x ,Y,w,h)
   