# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
from calculations import chessboard
import cv2
from scene.scene import Scene
from scene import edgeDetector, zone
from scene.qubic import QubicObject
import sys,os

from drawings.Draw import getColors
from func.trackFeatures import threshold
from calculations.chessboard import ChessboardDetector

class edgeDetectionTest(unittest.TestCase):

    writepath = ''
    
    def setUp(self):
        np.set_printoptions(precision=4)

    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def loadImage(self, filename, factor=1):
        print(filename)
        imgT = cv2.imread(filename)
#         factor = 0.25
        shape = (round(factor * imgT.shape[1]), round(factor * imgT.shape[0]))
        imgMap = np.empty(shape, dtype='uint8')
        imgMap = cv2.resize(imgT, imgMap.shape)
        scene = Scene(imgMap)
        return scene
    
    def getPoints(self,filename):
        factor = 1
        scene = self.loadImage(filename, factor)
        
        cd = ChessboardDetector()
        
        corners, z = cd.find_potential_corners(scene)
        
        offset = (z.offsetX,z.offsetY)
        print 'offset',offset
        corners2Shifted = cd.getZoneCorners(corners,offset)
        
        f = self.writepath+ 'tmp_chessboard.jpg'
        cv2.imwrite(f,z.image)
        
        gray = cv2.imread(f,flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)

        finalPoints,finalWorldPoints = cd.getPoints(corners2Shifted, gray)
        
        print finalPoints.tolist()
        print finalWorldPoints.tolist()
         
    def rrun(self,folder,i):
        self.folder = folder
        self.i = i
        self.writepath = '../img/results/automated/%d/' % folder
        
        filename = self.writepath+'calibration/src/%d/1.jpg' % (self.i)
        self.getPoints(filename)
        
        filename = self.writepath+'calibration/src/%d/2.jpg' % (self.i)
        self.getPoints(filename)
        
        
    def test_10_3(self):
        self.rrun(10,7)
        
#     def test_10_4(self):
#         self.rrun(10,4)   
        
       
       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    