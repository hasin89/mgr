# -*- coding: utf-8 -*-
'''
Created on Oct 31, 2014

@author: Tomasz
'''
import cv2
import numpy as np
from analyticGeometry import convertLineToGeneralForm
from zone import Zone

import func.markElements as mark


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
    
    # progowa dlugosc lini
    h_treshold = None
    
    # odwrotnosc minimalnej czesci długości obrazu np. 2 oznacza 50%, 3 oznacza 33% = 1/3
    part = 3.5
        
    def __init__(self,scene):
        
        self.scene = scene
        self.origin = scene.view.copy()
        
        self.edges_mask = self.findEdges(self.scene)
        
        self.mirror_line_Hough = None
        self.mirror_line = self.findMirrorLine(self.edges_mask)
        
        self.middle = self.calculateLineMiddle()
        self.mirrorZone = None
        
    
    def getReflectedZone(self,mirror_line):
        (x,y) = self.calculateLineMiddle()
        reflected = Zone(self.scene.view,0,0,self.scene.width,y)
        
        return reflected
    
    def getDirectZone(self,mirror_line):
        (x,y) = self.calculateLineMiddle()
        direct = Zone(self.scene.view,0,y,self.scene.width,self.scene.height-y)
        
        return direct
    
    def calculateLineMiddle(self):
        x = int(self.scene.width/2)
        y = int ( round (abs((self.mirror_line[0]*x+self.mirror_line[2])/self.mirror_line[1])))
        
        return (x,y)
    
    def calculatePointOnLine(self,x):
        x = int(x)
        y = int ( round (abs((self.mirror_line[0]*x+self.mirror_line[2])/self.mirror_line[1])))
        
        return (x,y)

    def findEdges(self,scene):
        '''
            find edges mask necessary to find the mirror line
        '''
        
        print 'find edges mask'
        
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
        
        print 'looking for the mirror line'
        mask = edges_mask
        
        part = self.part
        
        rho = self.rho
        theta = self.theta
        
        threshold=int(mask.shape[1]/part)
        
        self.h_treshold = threshold
        #znaldz linie hougha
        lines2 = cv2.HoughLines(mask,rho,theta,threshold)
        
        # znajdź linie o najmniejszym parametrze A (najbliższym poziomego) A=0 idealnie pozioma linia. max odchylenie 22.5
        Amin = 2
        mirror_line = None
        mirror_line_Hough = None
        
        for (rho,theta) in lines2[0][:7]:
            
            line = convertLineToGeneralForm((rho,theta),mask.shape)
            A = abs((round(line[0],0)))
            if A<Amin:
                mirror_line = line
                mirror_line_Hough = (rho,theta) 
                Amin=A 
        
        if mirror_line == None:
            print 'detected lines:'
            print lines2[0]
            
            if len(lines2[0])>0:
                mark.drawHoughLines(lines2[0], self.scene.view, (128,0,128), 5)
                self.scene.view[mask == 1] = (255,0,0)
                                    
            raise Exception("Mirror line not found!")
        self.mirror_line_Hough = mirror_line_Hough
        return mirror_line
    
    
    def findMirrorZone(self):
        
        shift = 100
        
        left, right = self.__findVerticalEdges(shift)
        
        if left == 0 or right == self.scene.width:
            shift = shift + 5
            left, right = self.__findVerticalEdges(shift)
        print 'sides',left,right   
        theta = 0
        mark.drawHoughLines([(left,theta),(right,theta)], self.scene.view, 255, 5)
        
        mirrorZone = Zone(self.scene.view,left,0,right-left,self.scene.height)
        self.mirrorZone=  mirrorZone
        
        return self.mirrorZone
    
    def __findVerticalEdges(self,shift,chess=False):
        #narysuj kreske na pustym obrazie
        canvas = np.zeros_like(self.edges_mask)
        chess = True
        if chess:
            
            
            rho = 1
            theta = np.pi/180
            threshold= int(self.edges_mask.shape[0]/4)
            mask = self.edges_mask.copy()
            mask[self.middle[1]:,:] = 0
            
            
            k = 10
            kernel = np.ones((k,k))
            mask = cv2.dilate(mask,kernel)
            mask = cv2.erode(mask,kernel)
            
            mask2 = cv2.distanceTransform(mask,3,3)
            tres = self.distanceTreshold * 4
            
            mask3 = np.where(mask2>tres,1,0).astype('uint8')
            k = 50
            kernel = np.ones((k,k))
            mask5 = cv2.dilate(mask3,kernel)
            
            mask5 = cv2.subtract(mask,mask5).astype('uint8')
            mask4 = np.where(mask5>0,255,0)
            
            lines2 = cv2.HoughLines(mask5,rho,theta,threshold)
            
            lines = []
            for l in lines2[0]:
                if l[1]>np.pi*0.25 and l[1]<np.pi*0.75:
                    continue
                lines.append(l)
            
            if lines is not None:
                mark.drawHoughLines(lines,self.scene.view,(0,0,255),5)
                canvas2 = canvas.copy()
                #vertical line
                mark.drawHoughLines(lines, canvas, 1, 1)
                #horizontal line
                mark.drawHoughLines([(self.mirror_line_Hough[0],self.mirror_line_Hough[1])], canvas2, 1, 1)
                
                output = canvas*canvas2
                
            
                
#             left,right = 0,self.scene.width
        else:
            mark.drawHoughLines([(self.mirror_line_Hough[0]-shift,self.mirror_line_Hough[1])], canvas, 1, 5)
            mark.drawHoughLines([(self.mirror_line_Hough[0]-shift,self.mirror_line_Hough[1])], self.scene.view, 1, 5)
            
            output = canvas*self.edges_mask
            
        o1 = np.nonzero(output)
        o2 = np.transpose(o1)
        
        #debuging purpose
        for o3 in o2:
            cv2.circle(self.scene.view, (o3[1],o3[0]), 10, (255,0,0), -1)

        right_side = np.where( (o1[1]-self.middle[0])>0, o1 , 999999 )
        
        if len(right_side[1])>0 and min(right_side[1])<999999:
            right = min(right_side[1])
        else:
            right = self.scene.width
        
        left_side = np.where( (self.middle[0] - o1[1]-1 )>0, o1 , 0 )
        if len(left_side[1])>0 and max(left_side[1])<999999:
            left = max(left_side[1])
        else:
            left = 0
        
        return left,right
        
#         
