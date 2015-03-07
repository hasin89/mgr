# -*- coding: utf-8 -*-
#/usr/bin/env python
'''
Created on Oct 31, 2014

@author: Tomasz
'''
import cv2
import numpy as np


class edgeDetector(object):
    
    
        
    def __init__(self,image,gauss_kernel=5):
        gauss_kernel = 5
        
#         image = self.gammaCorection(image)
        self.blur = cv2.GaussianBlur(image, (gauss_kernel, gauss_kernel), 0)
        self.image = image
        
        
    def gammaCorection(self,gray):
        gamma_correction = 1.55
        img_tmp = gray / 255.0
        cv2.pow(img_tmp, gamma_correction, img_tmp)
        gamma = img_tmp * 255.0
        gamma = gamma.astype('uint8')
        return gamma
        

    def SobelChanel(self,color):
        image = self.blur
        
        if color == 'R':
            chanel = 2
        elif color == 'G':
            chanel = 1
        elif color == 'B':
            chanel = 0
        else:
            chanel = 0
            
        #progressive processing x and y
        pro2 = cv2.Sobel(image[:,:,chanel],-1,0,1,ksize=3,delta=0)
        pro3 = cv2.Sobel(image[:,:,chanel],-1,1,0,ksize=3,delta=0)
        pro0 = self.combineSobel(pro2, pro3)
        
        #regressive processing x and y
        fl = cv2.flip(image,-1)
        reg2 = cv2.Sobel(fl[:,:,chanel],-1,0,1,ksize=3,delta=0)
        reg3 = cv2.Sobel(fl[:,:,chanel],-1,1,0,ksize=3,delta=0)
        reg0_fl = self.combineSobel(reg2, reg3)
        reg0 = cv2.flip(reg0_fl,-1)
        
        result = cv2.add(pro0,reg0)
        result = np.where(result<35,0,result)
        
        return result
    
    def combineSobel(self,A,B):
        '''
            calculates
            sqrt(A^2+B^2)
            
            probaably not used
        '''
        
        A = np.asarray(A,dtype='int64')
        B = np.asarray(B,dtype='int64')
        
        a2 = pow(A,2)
        b2 = pow(B,2)
        
        sumAB = a2+b2
        result = np.zeros_like(a2)
        np.sqrt(sumAB,result)
#         copied from GIMP
#         result = result*2
        return result
    
    def getSobel(self):
        
        r0 = self.SobelChanel('R')
        g0 = self.SobelChanel('G')
        b0 = self.SobelChanel('B')
        
        final = cv2.add(b0,g0)
        final = cv2.add(final,r0)
        
        mask = np.where(final>0,1,0).astype('uint8')
        
        return mask