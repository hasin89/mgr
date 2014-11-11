#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import numpy as np

class LabelFactory(object):
    
    #tabela korzeni
    P = {}
    
    #tablica etykiet
    L = None
    
    def __init__(self,binary):
        binary[0,:] = 0
        binary[-1,:] = 0
        binary[:,0] = 0
        binary[:,-1] = 0
        self.binary = binary
        
        self.currentLabel = 1
        self.L = np.zeros_like(binary)
        
    def run(self,binary):
        currentLabel = 1
        L = np.zeros_like(binary)
        P = {}
        
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
                currentLabel + 1
                self.L[e] = currentLabel
                self.P[currentLabel] = currentLabel
        
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
            
            #b
            if b == 1:
                self.copy(pe, pb)
            #c
            elif c == 1:
                #a
                if a == 1:
                    self.copy2(pe, pc, pa)
                #d
                elif d == 1:
                    self.copy2(pe, pc, pd)
                else:
                    self.copy(pe, pc)
            
            #a
            elif a == 1:
                self.copy(pe, pa)    
            
            #d
            elif d == 1:
                self.copy(pe, pd)
            
            else:
                self.newLabel(pe)
            
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
        
        pass
    
    def flatten(self):
        for i in range(len(self.P)):
            self.P[i] = self.P[self.P[i]]
            
    #decision tree
    
    def copy(self,currentPoint,point):
        self.L[currentPoint] = self.L[point]
        
    def copy2(self,currentPoint,point1,point2):
        self.L[currentPoint] = self._union(self.L[point1], self.L[point2])
        
        