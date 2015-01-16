'''
Created on Jan 8, 2015

@author: Tomasz
'''
import cv2
import numpy as np
import func.analise as an

class Contour(object):
    '''
    classdocs
    '''


    def __init__(self,label,cntMap):
        '''
        Constructor
        '''
        
        self.map = cntMap
        self.label = label
        
        indexes = np.where(cntMap == 1)
        points = np.transpose(indexes)
        orderedPoints = self.__getOrderedList(points)
        self.points = orderedPoints
        self.begining = orderedPoints[0]
        self.end = orderedPoints[-1]
        
        self.wayPoint = None
        self.polygon = self.aproximate() 
        
        
        self.lines = None
        
    def __getOrderedList(self,points):
        contour = []
        stack = map(tuple,list(points))
        
        start = stack[0]
        p1 = start
        y,x = p1
        positions = [ 
            (y-1,x-1),
            (y-1,x),
            (y-1,x+1),
            (y,x-1),
            (y,x+1),
            (y+1,x-1),
            (y+1,x),
            (y+1,x+1)
            ]
        begining = []
        for p in positions:
            if p in stack:
                i = stack.index(p)
                p2 = stack.pop(i)
                begining.append(p2)
        if len(begining) == 2:
            contour.insert(len(contour),begining[0])
            contour.insert(0,begining[1])
        elif len(begining) ==1:
            contour.append(begining[0])
            begining.append(None)
        
        p1 = begining[0]
        while len(stack)>0:
            y,x = p1
            positions = [ 
                (y-1,x-1),
                (y-1,x),
                (y-1,x+1),
                (y,x-1),
                (y,x+1),
                (y+1,x-1),
                (y+1,x),
                (y+1,x+1)
                ]
            counter = 0
            for p in positions:
                if p in stack:
                    i = stack.index(p)
                    p2 = stack.pop(i)
                    contour.append(p2)
                    p1 = p2
                else:
                    counter = counter+1
            if counter == 8:
                break
            
        if begining[1] is not None:
            p1 = begining[1]
            while len(stack)>0:
                y,x = p1
                positions = [ 
                    (y-1,x-1),
                    (y-1,x),
                    (y-1,x+1),
                    (y,x-1),
                    (y,x+1),
                    (y+1,x-1),
                    (y+1,x),
                    (y+1,x+1)
                    ]
                counter = 0
                for p in positions:
                    if p in stack:
                        i = stack.index(p)
                        p2 = stack.pop(i)
                        contour.insert(0,p2)
                        p1 = p2
                    else:
                        counter = counter+1
                if counter == 8:
                    break
            
        return contour
       
    def getLines(self):
        #znaldz linie hougha
        rho = 1
        theta = np.pi/180
        threshold = 20
        
        if (self.wayPoint is not None):
            points = self.points
            linesResult = np.array([])
            counter = 0
            for i in range(len(self.wayPoint)):
                far = (self.wayPoint[i][0][0],self.wayPoint[i][0][1])
                print far
                mapI = np.zeros_like(self.map)
                
                idx = points.index(far)
                if idx == 0 :
                    continue
                print 'index',idx
                
                indexes = map(np.array,np.transpose(np.array(points[:idx])))
                points = points[idx:]
                mapI[indexes] = 1
                
                linesI = cv2.HoughLines(mapI,rho,theta,threshold)
                
                if linesI is not None and len(linesI[0])>0:
                    print 'lines size',len(linesI[0])
                    counter += 1
                    linesResult = np.append(linesResult, linesI[0][0])
            print 'size',len(self.wayPoint)
            linesR = np.reshape(linesResult,(counter,2))
            print linesR
            self.lines = linesR
            
        else:
                    
            lines = cv2.HoughLines(self.map,rho,theta,threshold)
            self.lines = lines
            
            if lines is not None and len(lines[0])>0:
                self.lines = lines[0]
    #             mark.drawHoughLines(lines[0][:3], image6, (128,0,128), 1)
            else:
                self.lines = None

    def aproximate(self):
        start = self.begining
        tresh = 30
        polygon = cv2.approxPolyDP(np.asarray(self.points),tresh, False)
        if max > 30:
            self.wayPoint = polygon
        return polygon
    
    def findCornersOnContour(self,size):
        '''
            size - dlugosc probierza, offset pomiedzy elementami konturu dla ktorych zrobiony jest odcinek
        '''
    
        contour = self.contour
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