# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
from calculations import triangulation, DecompPMat, labeling
import cv2
import func.markElements as mark
from scene.mirrorDetector import mirrorDetector
from scene.zone import Zone

from scene.objectDetector2 import objectDetector2

from drawings.Draw import getColors

from calculations.labeling import LabelFactory
from drawings.Draw import getColors

from scene import analyticGeometry
from scene import edgeDetector

from skimage import morphology
from skimage import measure

import func.analise as an
from math import sqrt
from scene.wall import Wall
from scene.qubic import QubicObject
import calculations
from scene.LineDectecting import LineDetector

from math import cos
from collections import Counter
from func.markElements import corners
import pickle

class VertexesTest(unittest.TestCase):
    
    POINTS = []

    def setUp(self):
        np.set_printoptions(precision=4)

    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def getZones(self,folder,pic):
        factor = 1
        filename = '../img/%d/%d.JPG' % (folder, pic)
        scene = self.loadImage(filename, factor)
        
        md = mirrorDetector(scene)
        md.findMirrorZone()
        
        od = objectDetector2(md,scene.view)
        zoneA,zoneB,zoneC,zoneD = od.detect()
        
        return scene,zoneA,zoneB,zoneC,zoneD
       
    def loadZone(self,folder,pic,letter):
        factor = 1
        filename = '../img/results/automated/%d/objects2/%d_objects_on_mirror_%s.jpg' % (folder,pic,letter)
        zone = self.loadImage(filename, factor)
        if folder ==9 and pic == 1 and letter == 'B':
            self.POINTS = [
                           (2544,691),
                           (2622,624),
                           (2760,671),
                           (2684,740),
                           (2748,806),
                           (2533,827),
                           (2672,877)
                           ]
            
        
        return zone
     
       
    def proceed(self,zone,mask,folder, i, letter):
        pic = i
        
        qubic = QubicObject(zone)
        
        #caly bialy
        image = qubic.emptyImage.copy()
        image3 = image.copy()
        
        for kk,wall in qubic.walls.iteritems():
            
            #zaznaczenie powieszhni ściany           
            image3[wall.map == 1] = (255,255,255)   
                
        for vv in qubic.vertexes:
            cv2.circle(image3,(vv[0],vv[1]),1,(10,0,255),2) 
        if len(qubic.vertexes) > 7:
            raise Exception('too much vertexes')
        print 'wierzcholki: ', qubic.vertexes
        fname = '../img/results/automated/%d/obj5/pickle_vertex_%d_%s.p' % (folder, pic,letter)
        f = open(fname,'wb')
        pickle.dump(qubic, f)
        f.close()
        
#         f = open(fname,'rb')
#         obj = pickle.load(f)
        
        f = '../img/results/automated/%d/obj5/%d_lines_%s.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, image3)
            
        return image,mask


    def execute(self, folder, pic):
        i = pic
        
        zoneA = self.loadZone(folder, pic,'A')
        zoneB = self.loadZone(folder, pic,'B')
        zoneC = self.loadZone(folder, pic,'C')
        zoneD = self.loadZone(folder, pic,'D')
        
        ed = edgeDetector.edgeDetector(zoneA.view)
        mask = ed.getSobel()
        imgA,maskA = self.proceed(zoneA,mask,folder, i,'A')
        
        ed = edgeDetector.edgeDetector(zoneB.view)
        mask = ed.getSobel()
        imgB,maskB = self.proceed(zoneB,mask,folder, i,'B')
        
        ed = edgeDetector.edgeDetector(zoneC.view)
        mask = ed.getSobel()
        imgC,maskC = self.proceed(zoneC,mask,folder, i,'C')
        
        ed = edgeDetector.edgeDetector(zoneD.view)
        mask = ed.getSobel()
        imgD,maskD = self.proceed(zoneD,mask,folder, i,'D')
        
        
    
#     def test_9_8(self):
#         self.execute(9, 8)
        
    def test_9_1(self):
        self.execute(9, 1)
        
    def test_9_2(self):
        self.execute(9, 2)
# #         
    def test_9_3(self):
        self.execute(9, 3)
            
    def test_9_7(self):
        self.execute(9, 7)
#          
#          
    def test_9_10(self):
        self.execute(9, 10)
         
    def test_9_11(self):
        self.execute(9, 11)           

#     def test_8_3(self):
#         self.execute(8, 3)
#         
#     def test_8_6(self):
#         self.execute(8, 6)   
#         
#     def test_8_7(self):
#         self.execute(8, 7) 
#     
#     def test_8_19(self):
#         self.execute(8, 19) 
# 
#     def test_8_77(self):
#         self.execute(8, 77)

#     def test_7_1(self):
#         self.execute(7, 1)   
#         
#     def test_7_2(self):
#         self.execute(7, 2) 
#     
#     def test_7_3(self):
#         self.execute(7, 3) 
# 
#     def test_4_1(self):
#         self.execute(4, 1)   
#         
#     def test_4_2(self):
#         self.execute(4, 2) 
#     
#     def test_4_3(self):
#         self.execute(4, 3)
#     
#     def test_4_4(self):
#         self.execute(4, 4)   
#         
#     def test_4_5(self):
#         self.execute(4, 5) 
#     
#     def test_4_6(self):
#         self.execute(4, 6)
#     
#     def test_4_7(self):
#         self.execute(4, 7)   
#         
#     def test_4_8(self):
#         self.execute(4, 8) 
#     
#     def test_4_9(self):
#         self.execute(4, 9)  
# 
#     def test_5_1(self):
#         self.execute(5, 1)   
#         
#     def test_5_2(self):
#         self.execute(5, 2) 
#     
#     def test_5_3(self):
#         self.execute(5, 3)
#     
#     def test_5_4(self):
#         self.execute(5, 4) 
#         
#     def test_5_6(self):
#         self.execute(5, 6)
#     
#     def test_5_7(self):
#         self.execute(5, 7)   
#         
#     def test_5_8(self):
#         self.execute(5, 8) 
#     
#     def test_5_9(self):
#         self.execute(5, 9)
#         
#     def test_5_10(self):
#         self.execute(5, 10) 
#     
#     def test_5_11(self):
#         self.execute(5, 11)
            
            
    def loadImage(self, filename, factor=1):
        print(filename)
        imgT = cv2.imread(filename)
#         factor = 0.25
        shape = (round(factor * imgT.shape[1]), round(factor * imgT.shape[0]))
        imgMap = np.empty(shape, dtype='uint8')
        imgMap = cv2.resize(imgT, imgMap.shape)
        from scene.scene import Scene
        scene = Scene(imgMap)
        return scene
    
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
