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
from contour import Contour

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
        
        self.emptyMask = np.zeros(self.image.shape[:2],dtype='uint8')
        
        edgeMask = self.getEdges()
        self.edgeMask = edgeMask
        
        walls, labelsMap, backgroundLabel, labels = self.findWalls(edgeMask)
        self.labelsMap = labelsMap
        self.labels = labels
        self.backgroundLabel = backgroundLabel
        self.walls = walls
         
        skeleton, edgeLabelsMap, edgeLabels, nodesPoints  = self.findContours()
        
        self.skeleton = skeleton
        self.edgeLabelsMap = edgeLabelsMap
        self.edgeLabels = edgeLabels
        
        nodes = self.getNodes(nodesPoints,skeleton)
        self.nodes = nodes
        
        self.getWallsProperties(walls,edgeLabels,edgeLabelsMap,nodes)
        
        
    
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
        return edgeMask
    
    def openOperation(self,res,label,kernelSize = 3):
        background = np.where(res == label ,255,0).astype('uint8')
        kernel = np.ones((kernelSize,kernelSize),np.uint8)
        background = cv2.dilate(background,kernel,iterations = 1)
        background = cv2.erode(background,kernel,iterations = 1)
        res = np.where(background == 255,label,res)
    
        return res
    
    def getNodes(self,points,binaryMap):
        nodes = {}
        
        lf = LabelFactory([])
        nodeMap = np.zeros_like(binaryMap)
        ll = map(np.array,np.transpose(np.array(points)))
        nodeMap[ll] = 1
        
        nodeLabelsMap = lf.getLabelsExternal(nodeMap, neighbors=8, background=0)
        nodeLabels =  np.unique(nodeLabelsMap)
        
        for nodeLabel in nodeLabels:
            if nodeLabel == -1:
                continue
            indexes = np.where(nodeLabelsMap == nodeLabel)
#             nodes[nodeLabel] = indexes 
            
            nmax = 0
            for p in np.transpose(indexes):
                y = p[0]
                x = p[1]
                submask = nodeMap[y-1:y+2,x-1:x+2]
                ones = np.nonzero(submask)
                neibours = ones[0].size
                if neibours>1:
                    if nmax<neibours:
                        nmax = neibours
                        N = (y,x)
            if nmax>0:
                nodes[nodeLabel] = N                
            
        return nodes
        
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
        
        #operacja otwarcia na tle - eliminacja dorbnych zaklucen
        labelsMap = self.openOperation(labelsMap, backgroundLabel, kernelSize=5)
        
        ei = self.emptyImage.copy()
        ei[labelsMap == backgroundLabel] = (255,255,255)

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

        edges = ContourDectecting.transfromEdgeMaskIntoEdges(self.edgeMask,self.emptyImage)
        
        ocd = ContourDectecting.ObjectContourDetector(edges)
        skeleton2, edgeLabelsMap, edgeLabels, nodesPoints = ocd.fragmentation(ocd.skeleton) 
        
        
        
        self.skeleton2 = skeleton2
        return skeleton2, edgeLabelsMap, edgeLabels, nodesPoints
    
    def getWallsProperties(self,walls,edgeLabels,edgeLabelsMap,nodes):
        '''
        area_dist - mapa dystansu od sciany
        labelsT - etykiety krawedzi
        resT - mapa krawedzi zetykietyzowana
        img - obraz ostateczny
        Bmap - empty map
        '''
        
        for kk,wall in walls.iteritems():
            #szukanie konturow nalezacych do scian
            for edge_label in edgeLabels:
                
                indexes = np.where(edgeLabelsMap == edge_label)
                values = wall.wallDistance[indexes]
                
                max_value = np.max(values)
                min_value = np.max(values)
                if max_value < 20 and min_value>1:
                    mmap = np.where(edgeLabelsMap == edge_label,1,0).astype('uint8')
                    c = Contour(edge_label,mmap)
                    
                    wall.contours.append(c)
                    wall.contoursDict[edge_label] = c
            #szukanie wezlow nalezacych do scian
            for kk,node in nodes.iteritems():
                values = wall.wallDistance[node]
                max_value = np.max(values)
                min_value = np.max(values)
                if max_value < 20 and min_value>1:
                    #node belong to the wall
                    wall.nodes.append(node)
                    
        return walls    