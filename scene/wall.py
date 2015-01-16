'''
Created on Jan 8, 2015

@author: Tomasz
'''
import cv2
import numpy as np
import func.analise as an

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
        self.analyzeDefects(defs)
        
        # map of the distances from the wall
#         wallInverted = np.where(labelsMap == label ,0,1).astype('uint8')
        wallInverted = area2#np.where(wallMap == 1 ,0,1).astype('uint8')
        self.wallDistance = cv2.distanceTransform(wallInverted,cv2.cv.CV_DIST_L1,3)
        
        self.contours = []
        self.contoursDict = {}
        self.nodes = []
        
        self.vertexes = []
        
    def analyzeDefects(self,defs):
        fars = []
        for kk, defects in enumerate(defs): 
                cnt = self.cnt
                for jj in range(defects.shape[0]):
                    s,e,f,d = defects[jj]
                    
                    #line endings
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    
                    #defect point the furthest from the line
                    far = tuple(cnt[f][0])
                    line = an.getLine(start, end,0)
                    
                    distance = an.calcDistFromLine(far, line)
                    
                    if distance > 20:
                        print 'wielobok wypukly. Punkt:',far
                        fars.append(far)
        if len(fars)>0:
            self.convex = (True,fars)
        else:
            self.convex = (False,fars)
    
    
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