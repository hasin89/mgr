'''
Created on Jan 8, 2015

@author: Tomasz
'''
import cv2
import numpy as np

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
        self.contour = cnt
        
        #obrys scany
        hull = cv2.convexHull(cnt,returnPoints = False)
        self.convexHull = hull
        
        #defect obrysu
        defs = cv2.convexityDefects(cnt,hull)
        self.hullDefects = defs
        
        # map of the distances from the wall
#         wallInverted = np.where(labelsMap == label ,0,1).astype('uint8')
        wallInverted = area2#np.where(wallMap == 1 ,0,1).astype('uint8')
        self.wallDistance = cv2.distanceTransform(wallInverted,cv2.cv.CV_DIST_L1,3)
    