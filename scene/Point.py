# -*- coding: utf-8 -*-
'''
Created on Sep 4, 2014

@author: Tomasz
'''
import numpy as np 


class Point(object):
    '''
    classdocs
    '''
    POINT_TYPE_CONTOUR = 1
    POINT_TYPE_VERTEX = 0
    POINT_TYPES = {
             0:'VERTEX',
             1:'CONTOUR'
             }


    def __init__(self,x,y,origin,pointType = None):
        '''
        Constructor
        '''
        self.x = x
        self.y = y
        self.origin = origin
        self.type = pointType
        
    def __str__(self):
        return "p( %d , %d )" % (self.x,self.y)

    def __repr__(self):
        return self.__str__()
    
    def getNeibours(self):

        yMax,xMax = self.origin.shape
        x = self.x
        y = self.y
    
        check = {}
    
        check['a'] = (x+1,y)
        check['c'] = (x-1,y)
        check['e'] = (x+1,y+1)
        check['g'] = (x-1,y-1)
    
        check['b'] = (x, y+1)
        check['d'] = (x,y-1)
        check['f'] = (x-1,y+1)
        check['h'] = (x+1,y-1)
    
    
        if x == 0:
            del check['c']
            del check['f']
            del check['g']
    
        elif x == xMax-1:
            del check['a']
            del check['e']
            del check['h']
    
        if y == 0:
            del check['d']
            if 'h' in check.keys():
                del check['h']
            if 'g' in check.keys():
                del check['g']
    
        elif y == yMax-1:
            del check['b']
            if 'e' in check.keys():
                del check['e']
            if 'f' in check.keys():
                del check['f']
    
        for value in check.itervalues():
            if self.origin[value[1],value[0]] == 255:
                return (value[0],value[1])
    
        return False
    
    def searchNearBy(self):
        u"""
        Szukaj wkoło punktów konturu
        
        zwraca punkt i dystans
        """
        print 'search'
        edge = self.origin
        x = self.x
        y = self.y
        sp = edge.T.shape
    
        rangeSizes = range(5,19,2)
        for i in rangeSizes:
            mask = self.__generateRange(i)
            points = np.asarray((x,y))+mask
    
            #filtowanie z poza granic
            xx = points[:,0]
            yy =points[:,1]
            XoverflowIndex = np.where(xx>sp[0]-1)
            YoverflowIndex = np.where(yy>sp[1]-1)
            wrongIndexes = np.union1d(XoverflowIndex[0],YoverflowIndex[0])
            points= np.delete(points,wrongIndexes,0)
    
            p = [edge[points[k][1],points[k][0]] for k in range(len(points))]
            non = np.nonzero(p)[0]
            if non.size>0:
                no = np.nonzero(p)[0][0]
                return points[no] , i
    
        return np.asarray([]),False
    
    
    def __generateRange(self,maskSize):
    
        n = (maskSize-1)/2
        mask = []
        x=n
        y=n-1
        while x<n+1:
            while y<n:
                while x>-n:
                    while y>-n:
                        mask.append((x,y))
                        y-=1
                    mask.append((x,y))
                    x-=1
                mask.append((x,y))
                y+=1
            mask.append((x,y))
            x+=1
    
        return mask