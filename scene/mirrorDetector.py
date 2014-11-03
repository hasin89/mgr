# -*- coding: utf-8 -*-
'''
Created on Oct 31, 2014

@author: Tomasz
'''
import cv2
import numpy as np
from analyticGeometry import convertLineToGeneralForm
from zone import Zone



class mirrorDetector(object):
    
    # blur
    gauss_kernel = 5
    
    constant = 5
    blockSize = 101
    tresholdMaxValue = 255
    
    # prog dla grubej kreski
    distanceTreshold = 4
    
    # parametry wykrawacza lini
    
    # wielkosc akumulatora
    rho = 1.5
    # theta = 0.025
    # rozdzielczosc kata
    theta = np.pi/180
    # odwrotnosc minimalnej czesci długości obrazu np. 2 oznacza 50%, 3 oznacza 33% = 1/3
    part = 3
        
    def __init__(self,scene):
        
        self.scene = scene
        
        self.edges_mask = self.findEdges(self.scene)
        
        self.mirror_line = self.findMirrorLine(self.edges_mask)
        
    
    def getReflectedZone(self,mirror_line):
        
        y = abs((mirror_line[0]*self.scene.view.shape[0]/2.0+mirror_line[2])/mirror_line[1])
        reflected = self.scene.view[:y, :]
        
        reflected = Zone(self.scene.view,0,0,self.scene.width,y)
        
        return reflected
    
    def getDirectZone(self,mirror_line):
        y = abs((mirror_line[0]*self.scene.view.shape[0]/2.0+mirror_line[2])/mirror_line[1])
        
        direct = self.scene.view[y:, :]
        
        direct = Zone(self.scene.view,0,y,self.scene.width,self.scene.height-y)
        
        return direct

    def findEdges(self,scene):
        '''
            find edges mask necessary to find the mirror line
        '''
        
        print 'horizontal division'
        
        gauss_kernel = self.gauss_kernel
        tresholdMaxValue = self.tresholdMaxValue
        blockSize = self.blockSize
        constant = self.constant
        
        distanceTreshold = self.distanceTreshold
        
        gray = scene.gray
        gray_filtred = cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtred,
                                             maxValue=tresholdMaxValue,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=blockSize,
                                             C=constant)
        
        
        
        # CV_DIST -> sasiedztwo 8 spojne
        dst = cv2.distanceTransform(edge_filtred,cv2.cv.CV_DIST_C,3) # 3 to jest wielkość maski, która nie ma znaczenia dla 8 spojnego sasiedztwa
        
        # znajdz punkty o odleglosci wiekszej niz prog. generalnie grube krechy
        mask = np.where(dst>distanceTreshold,1,0).astype('uint8')
        
        mask2 = np.where(dst>distanceTreshold,255,0).astype('uint8')
        f = '../img/results/matching/__test__.jpg'
        cv2.imwrite(f,mask2)
        
        return mask
        
        
    def findMirrorLine(self,edges_mask):         
        '''
        function responsible for division of the direct and reflected image
        '''
        
        mask = edges_mask
        
        part = self.part
        
        rho = self.rho
        theta = self.theta
        
        threshold=int(mask.shape[1]/part)
        
        
        #znaldz linie hougha
        lines2 = cv2.HoughLines(mask,rho,theta,threshold)
        
        # znajdź linie o najmniejszym parametrze A (najbliższym poziomego) A=0 idealnie pozioma linia. max odchylenie 22.5
        Amin = 2
        mirror_line = None
        
        for (rho,theta) in lines2[0][:2]:
            print (rho,theta)
            line = convertLineToGeneralForm((rho,theta),mask.shape)
            A = abs((round(line[0],0)))
            if A<Amin:
                mirror_line = line
                Amin=A 
        
        if mirror_line == None:
            raise Exception("Mirror line not found!")
        
        return mirror_line