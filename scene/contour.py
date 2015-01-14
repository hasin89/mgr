'''
Created on Jan 8, 2015

@author: Tomasz
'''
import cv2
import numpy as np

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
            print 'not none'
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
        rho = 2
        theta = np.pi/90
        threshold = 20
                
        lines = cv2.HoughLines(self.map,rho,theta,threshold)
        
        self.lines = lines
        
        if lines is not None and len(lines[0])>0:
            self.lines = lines[0]
#             mark.drawHoughLines(lines[0][:3], image6, (128,0,128), 1)
        else:
            self.linse = None
