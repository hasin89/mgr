# -*- coding: utf-8 -*-
'''
Created on Sep 4, 2014

@author: Tomasz
'''
import numpy as np 


class Contour(object):
    '''
    classdocs
    '''

    def __init__(self,Id,origin):
        '''
        Constructor
        '''
        self.id = Id
        self.points = []
        self.origin = origin
        self.length = None
        
    def AddPoint(self,point):
        self.points.append(point)
        self.length = len(self.points)
        
    def JoinContour(self,contour):
        if isinstance(contour, Contour):
            self.points.extend(contour.points)
        else:
            raise TypeError
    
    def GetPointsArray(self):
        return [(p.x,p.y) for p in self.points]