# -*- coding: utf-8 -*-
'''
Created on Jan 8, 2015

@author: Tomasz
'''
import cv2
import numpy as np
import func.analise as an
from func import objects as obj

class Wall(object):
    '''
    classdocs
    '''


    def __init__(self,label,wallMap,area2):
        '''
        Constructor
        '''
        
        self.map = wallMap
        self.label = label
        
        #kontur sciany        
        cnts = cv2.findContours(wallMap,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        cnt = cnts[0][0]
        self.cnt = cnt
        
        #obrys scany
        self.convex = None
        hull = cv2.convexHull(cnt,returnPoints = False)
        self.convexHull = hull
        
        #defect obrysu
        defs = cv2.convexityDefects(cnt,hull)
        self.hullDefects = defs
        self.convex = self.analyzeDefects(defs,cnt)
        
        # map of the distances from the wall
#         wallInverted = np.where(labelsMap == label ,0,1).astype('uint8')
        wallInverted = area2#np.where(wallMap == 1 ,0,1).astype('uint8')
        self.wallDistance = cv2.distanceTransform(wallInverted,cv2.cv.CV_DIST_L1,3)
        
        self.contours = []
        self.contoursDict = {}
        self.nodes = []
        
        self.vertexes = []
        
    def analyzeDefects(self,defs,cnt,treshold = 20):
        fars = []
        lines = []
        for kk, defects in enumerate(defs): 
                
                for jj in range(defects.shape[0]):
                    s,e,f,d = defects[jj]
                    
                    #line endings
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    
                    #defect point the furthest from the line
                    far = tuple(cnt[f][0])
                    line = an.getLine(start, end,0)
                    
                    distance = an.calcDistFromLine(far, line)
                    
                    if distance > treshold:
                        print 'wielobok wypukly. Punkt:',far
                        fars.append(far)
                        lines.append((start, end))
        if len(fars)>0:
            return (True,fars,lines)
        else:
            return (False,fars,lines)
    
    
    def findPotentialCorners(self):
        points = []
        corners = []
        for x in self.cnt:
            point = tuple(x[0])
            points.append(point)
        indexes = self.__findCornersOnContour(points, 50)
        for ii in indexes:
            corners.append(points[ii])
            
        return corners

    
    def __findCornersOnContour(self,contour,size):
        '''
            size - dlugosc probierza, offset pomiedzy elementami konturu dla ktorych zrobiony jest odcinek
        '''
    
        if len(contour)>size:
            indexes = []
            dist = an.calcDistances(contour,size,int(size*0.1))
    
            for d in dist.iterkeys():
                segment1 = dist[d]
                MaxValue = max(np.asarray(segment1)[:,1])
                index = np.where(segment1 == MaxValue)[0]
                indexes.append(segment1[index[0]][0])
            return indexes
        else:
            return []
        
        
    def getLinesCrossings(self):
        '''
        zwraca punkty bedące przecięciami podanych prostych
    
        lines - lista prostych będących krotkami [ (a,b,c) , ... ]
        edge - płótno
    
        return [ (x,y) ] - lista puntków będacych przecieciami
        '''
        lines = []
        shape = self.map.shape
        boundaries = np.array([[0,0],[shape[1],0],[shape[1],shape[0]],[0,shape[0]]])
        
        contours = self.contours
        for c in contours:
            lines.extend(c.lines)
        
        linesGeneral = []
    
        for (rho, theta) in lines:
            # blue for infinite lines (only draw the 5 strongest)
            a,b,c = an.convertLineToGeneralForm((rho,theta),shape)
            linesGeneral.append((a,b,c))
            
        
        crossing = []
        triple = {}
        fars = self.convex
        
        for k in linesGeneral:
            count = 0
            for l in linesGeneral:
                if k == l:
                    continue
                
                p = an.get2LinesCrossing(k,l)
                # sprawdź czy leży wewnątrz strefy
                if p != False:
                    isinside = cv2.pointPolygonTest(boundaries,p,0)
                    if isinside>0:
                        dist = self.wallDistance[p[1],p[0]]
                        if dist<30:
                            if p != False:
                                count += 1
                                #jesli ten punkt lezy blisko punktu wkleslosci to linia bedzie miala 3 przeciecia,
                                #ale to jest ok, wiec nie licz tego punktu
                                if len(fars[1]) > 0:
                                    for f in fars[1]:
                                        dist = an.calcLength(p, f)
                                        if dist < 20:
                                            count -= 1
                                
                                if p in crossing:
                                    
                                    continue
                                crossing.append(p)
                                
                    else:
                        pass
            if count > 2:
#                 print 'potrojny'
                #zabierz trzy ostatnie punkty przeciec
                triple[k] = crossing[-count:]
            
#         print 'triples:',triple
        #jeśli są jakieś linie z 3 punktami przeciecia, co niepowinno sie zdarzyc
        if len(triple) > 0 and fars[0] and False:
            
            # dla lini zawierajacych 3 punkty przeciecia i nie jest to punkt wkleslosci        
            keys = [k for k,v in triple.iteritems() if len(v)>2]
            for k in keys:
                points = triple[k]
                #wywal wszystkie punkty tej lini z listy przeciec jesli tam sa
                for p1 in points:
                    if p1 in crossing:
                        crossing.remove(p1)
                pair = an.getMostSeparatedPointsOfLine(points)
                    
                #dwa najbardziej od siebie odlegle punkty dodaj do listy przeciec
                for p in pair:
                    if p in crossing:
                        continue
                    else:
                        crossing.append(p)
    
        print 'crossings :', crossing
        return crossing, fars#,poly,vertexes