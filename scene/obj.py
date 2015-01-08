'''
Created on Jan 8, 2015

@author: Tomasz
'''
import numpy as np
import edgeDetector
import cv2
from calculations.labeling import LabelFactory
from wall import Wall
import ContourDectecting

class QubicObject(object):
    '''
    klasa zawierajaca obiekt szescianu
    '''


    def __init__(self,objectZone):
        '''
        Constructor
        '''
        self.image = objectZone.view
        emptyImage = self.image.copy()
        emptyImage[:] = (0,0,0)
        self.emptyImage = emptyImage
        
        edgeMask = self.getEdges()
        self.edgeMask = edgeMask
        
        walls, labelsMap, backgroundLabel, labels = self.findWalls(edgeMask)
        
        
        self.labelsMap = labelsMap
        self.labels = labels
        self.backgroundLabel = backgroundLabel
         
        self.walls = walls
         
        skeleton2, edgeLabelsMap, edgeLabels, nodes  = self.findContours()
        self.skeleton2 = skeleton2
        
    
    def getEdges(self):
        '''
            zwraca krawdedzie wykryte kolorowym operatorem Sobela
        '''
#         gauss_kernel = 5
#         img = cv2.GaussianBlur(self.image, (gauss_kernel, gauss_kernel), 0)
        ed = edgeDetector.edgeDetector(self.image)
        edgeMask = ed.getSobel()
        
        ei = self.emptyImage.copy()
        ei[:] = (0,0,0)
        ei[edgeMask > 0] = (255,255,255)
        f = '../img/results/automated/9/objects2/debug/skeleton_sobel_2.jpg' 
        print 'savaing to ' + f
        cv2.imwrite(f, ei)
        
        return edgeMask
    
    def openOperation(self,res,label,kernelSize = 3):
        background = np.where(res == label ,255,0).astype('uint8')
        kernel = np.ones((kernelSize,kernelSize),np.uint8)
        background = cv2.dilate(background,kernel,iterations = 1)
        background = cv2.erode(background,kernel,iterations = 1)
        res = np.where(background == 255,label,res)
    
        return res
    

        
    
    def findWalls(self,edgeMask):
        '''
            znajduje sciany i tlo
            rpzeprowadza operacje otwarcia - zaktualizowane krawedzie
        '''
        
        lf = LabelFactory([])
        #etykietowani obszarow niebedacych oddzielonych krawedziami (scian)
        res = lf.getLabelsExternal(edgeMask, neighbors=8, background=1)
        
        #znalezienie tla
        labelsMap,labels,backgroundLabel = lf.getBackgroundLabel(res)
        
        ei = self.emptyImage.copy()
        ei[labelsMap == backgroundLabel] = (255,255,255)
        f = '../img/results/automated/9/objects2/debug/skeleton_background_2.jpg' 
        print 'savaing to ' + f
        cv2.imwrite(f, ei)
        
        #operacja otwarcia na tle - eliminacja dorbnych zaklucen
        labelsMap = self.openOperation(labelsMap, backgroundLabel, kernelSize=5)
        
        ei = self.emptyImage.copy()
        ei[labelsMap == backgroundLabel] = (255,255,255)
        f = '../img/results/automated/9/objects2/debug/skeleton_open_b_2.jpg' 
        print 'savaing to ' + f
        cv2.imwrite(f, ei)
        
        walls = {}
        
        #dla kazdej etykiety poza etykieta tla czyli dla kazdej sciany
        for label in labels:
            if label == backgroundLabel:
                continue
            #operacja otwarcia z duzym ziarnem/maska
            labelsMap = self.openOperation(labelsMap, label, kernelSize=9)
            
            #znajdz kontury sciany
            wallMap = np.where(labelsMap == label ,1,0).astype('uint8')
            area2 = np.where(labelsMap == label ,0,1).astype('uint8')
            w = Wall(label,wallMap,area2)
            
            walls[label] = w
        
        self.edgeMask = np.where(labelsMap == -1,1,0).astype('uint8')
        
        return walls, labelsMap, backgroundLabel, labels

    
    def findContours(self):
        ei = self.emptyImage.copy()
        ei[:] = (0,0,0)
        ei[self.edgeMask > 0] = (255,255,255)
        f = '../img/results/automated/9/objects2/debug/skeleton_to_skeleton_2.jpg' 
        print 'savaing to ' + f
        cv2.imwrite(f, ei)
        edges = ContourDectecting.transfromEdgeMaskIntoEdges(self.edgeMask,self.emptyImage)
        
        
        
        ocd = ContourDectecting.ObjectContourDetector(edges)
        skeleton2, edgeLabelsMap, edgeLabels, nodes = ocd.fragmentation(ocd.skeleton) 
        self.skeleton2 = skeleton2
        return skeleton2, edgeLabelsMap, edgeLabels, nodes    