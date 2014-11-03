'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
from calculations import triangulation,DecompPMat
import cv2
import func.trackFeatures as features
from scene.edge import edgeMap
import func.markElements as mark
from scene.analyticGeometry import convertLineToGeneralForm
from scene.mirrorDetector import mirrorDetector
from scene.zone import Zone

class LocalizationTest(unittest.TestCase):


    def setUp(self):
        np.set_printoptions(precision=4)


    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def testZone(self):
        factor = 1
        
        folder = 8
        i = 7 
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        scene = self.loadImage(filename,factor)
        y = 1200
        direct = Zone(scene.view,0,y,scene.width,scene.height-y)
        
        
    def testDivision(self):
        
        folder = 8
        i= 77
        i =7
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        print "testing division old and new style"
        
        factor = 1
        scene = self.loadImage(filename,factor)
        
        md = mirrorDetector(scene)
        md.findEdges(scene)
        md.findMirrorLine(md.edges_mask)
        direct = md.getDirectZone(md.mirror_line)
        
        direct.mask
        
            
    def loadImage(self,filename,factor = 1):
        print(filename)
        imgT = cv2.imread(filename)
#         factor = 0.25
        shape = (round(factor*imgT.shape[1]),round(factor*imgT.shape[0]))
        imgMap = np.empty(shape,dtype='uint8')
        imgMap = cv2.resize(imgT,imgMap.shape)
        from scene.scene import Scene
        scene = Scene(imgMap)
        return scene
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    