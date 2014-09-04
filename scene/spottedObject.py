# -*- coding: utf-8 -*-
'''
Created on Sep 3, 2014

@author: Tomasz
'''
import cv2
import numpy as np
from func import histogram
import func.trackFeatures as features
import Draw

class spottedObject(object):
    u'''
        reprezentuje wykryty obiekt 
    '''


    def __init__(self,CNT):
        '''
        Constructor
        '''
        self.CNT = CNT
        x,y,w,h = cv2.boundingRect(self.CNT)
        
        # SqrBnd zawiera  wielokatna obwiednie bryły ? a nie prostokątną?
        self.sqrBnd = (x,y,w,h)

    
    def draw(self,canvas,color):
        for i in range(0,len(self.CNT)-1):
            Draw.Segment(canvas,self.points[i] ,self.points[i+1])