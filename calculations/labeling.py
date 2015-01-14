#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import numpy as np
from drawings.Draw import getColors
from skimage import measure

class LabelFactory(object):
    
    #tabela korzeni
    P = {}
    
    #tablica etykiet
    L = None
    
    def __init__(self,binary=None):#,binary):
#         binary[0,:] = 0
#         binary[-1,:] = 0
#         binary[:,0] = 0
#         binary[:,-1] = 0
#         self.binary = binary
#         
#         self.currentLabel = 1
        if binary is not None:
            self.L = np.zeros_like(binary)
        pass
        
    
        
    def run2(self):
        
        
        padding = np.pad(self.binary,(1,2),'constant')
        
        aa = padding[:-3,:-3]
        bb = padding[:-3,1:-2]
        cc = padding[:-3,2:-1]
        dd = padding[1:-2,:-3]
        ee = padding[1:-2,1:-2]
        
        foreground = np.nonzero(self.binary)
        a1 = aa[foreground]
        b1 = bb[foreground]
        c1 = cc[foreground]
        d1 = dd[foreground]
        
        
        for a,b,c,d,y,x in np.nditer([a1,b1,c1,d1,foreground[0],foreground[1]]):
            
            pa = (y-1,x-1) 
            pb = (y-1,x)
            pc = (y-1,x+1)
            pd = (y,x-1)
            
            pe = (y,x)
            
            
            
    def newLabel(self,point):
        self.currentLabel = self.currentLabel + 1
        self.L[point] = self.currentLabel
        self.P[self.currentLabel] = self.currentLabel
    
    #union find operations
    
    def findRoot(self,label):
        root = label
        P = self.P
        while P[root] < root:
            root = P[root]
        return root
    
    def setRoot(self,label,root):
        P = self.P
        while P[label] < label:
            nextLabel = P[label]
            P[label] = root
            label = nextLabel
        P[label] = root
        self.P = P
            
    def find(self,label):
        root = self.findRoot(label)
        self.setRoot(label, root)
        return root
    
    def _union(self,label1,label2):
        root1 = self.findRoot(label1)
        if label1 != label2:
            root2 = self.findRoot(label2)
            if root1 > root2:
                root1 = root2
            self.setRoot(label2,root1)
        self.setRoot(label1,root1)
        return root1
            
    def flattenLabels(self):
        P = self.P
        k = 2
        for label in range(2,len(self.P)+2):
            if P[label] < label:
                j = P[label]
                P[label] = P[j]
            else:
                P[label] = k
                k = k + 1
        self.P = P
    
    def flatten(self):
        for i in range(2,len(self.P)+2):
            self.P[i] = self.P[self.P[i]]
            
    #decision tree
    
    def copy(self,currentPoint,point):
        self.L[currentPoint] = self.L[point]
        
    def copy2(self,currentPoint,point1,point2):
        self.L[currentPoint] = self._union(self.L[point1], self.L[point2])
        
    def run(self,binary):
        binary[0,:] = 0
        binary[-1,:] = 0
        binary[:,0] = 0
        binary[:,-1] = 0
        currentLabel = 1
        L = np.zeros_like(binary)
        
        
        copy = self.copy
        copy2 = self.copy2
        
        foreground = np.nonzero(binary)
        
        for y,x in np.nditer(foreground):
            
            a = (y-1,x-1) 
            b = (y-1,x)
            c = (y-1,x+1)
            d = (y,x-1)
            e = (y,x)
            
            #b
            if binary[b]:
                copy(e, b)
            #c
            elif binary[c]:
                #a
                if binary[a]:
                    copy2(e, c, a)
                #d
                elif binary[d]:
                    copy2(e, c, d)
                else:
                    copy(e, c)
            
            #a
            elif binary[a]:
                copy(e, a)    
            
            #d
            elif binary[d]:
                copy(e, d)
            
            else:
                # set current label
                currentLabel = currentLabel + 1
                self.L[e] = currentLabel
                self.P[currentLabel] = currentLabel
                
    def flattenMap(self):
        P = self.P
        L = self.L
        for k,v in P.iteritems():
            L = np.where(L == k,v,L)
        self.L = L
        
    def getPreview(self):
        
        self.flattenMap()
        
        L = self.L
        
        uni =  np.unique(L)
        colorSpace = np.zeros((L.shape[0],L.shape[1],3),dtype='uint8')
        colors = getColors(len(uni))
        
        tempspace1 = L.copy()
        for i in range(0,len(uni)):
            if uni[i] == 0:
                continue
            tempspace = np.where(tempspace1 == uni[i],1,0)
            ids = np.nonzero(tempspace)
            colorSpace[ids] = colors[i]
            
        return colorSpace
    
    def convert2ContoursForm(self):
        contours = {}
        L = self.L
        uni =  np.unique(L)
        for i in range(0,len(uni)):
            if uni[i] == 0:
                continue
            tempspace = np.where(L == uni[i],1,0)
            ids = np.nonzero(tempspace)
            
            li = np.transpose(ids)
            contours[i] = map(tuple,li)
        return contours
        
# funkcja liczaca powierzchnie etykiet otrzmnaych z innego framworku
    def getBackgroundLabel(self,res,treshold=50,add_to_biggest=True):
        '''
            zwraca mape etykiet, liste etykiet i etykiete tla
            filtruje etykiety mniejsze niz treshold i przypisuje im etykiete tla (jesli flaga == True)
        '''
        uni =  np.unique(res)
        
        counts = []
        for k,j in enumerate(uni):
            if j == -1:
                continue
            count = np.where(res == j,1,0)
            a = np.count_nonzero(count)
            # ignore meaningless labels
            if a < treshold:
                # save these points as background color
                res = np.where(res == j,-2,res)
                a = 0
            counts.append(a)
        
        labelsI = np.nonzero(counts)[0]
        
        # find the background label
        maxI = np.argmax(counts)
        
        # save these points as background color (now there is the background label number known)
        if add_to_biggest:
            res = np.where(res == -2,maxI,res)
        else:
            res = np.where(res == -2,-1,res)
        return res, labelsI, maxI
    
    def getLabelsExternal(self,mask,neighbors=8,background=0):
        '''
            dokonuje etykietyzacji za pomoca zewnetrzej biblioteki
        '''
        LabelsMap = measure.label(mask,neighbors,background)
        LabelsMap = np.asarray(LabelsMap)
        LabelsMap.reshape(LabelsMap.shape[0],LabelsMap.shape[1],1)
        
        return LabelsMap
    
    def colorLabels(self,image,labelsMap,background_label = -1):
        '''
        kolorowanie etykiet
        '''
        uni =  np.unique(labelsMap)
        colors = getColors(len(uni))
        for k,j in enumerate(uni):
            if j == background_label:
                continue
            color = np.asarray(colors[k])
            image[labelsMap == j] = color
        
        return image