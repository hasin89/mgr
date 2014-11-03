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
        edges = md.findEdges(scene)
        md.findMirrorLine(md.edges_mask)
        
        print "mirror line:"+str(md.mirror_line)
        direct = md.getReflectedZone(md.mirror_line)
        
        edges2 = np.where(direct.mask>0,edges,0).astype('uint8')
        edges2 = np.where(edges2>0,255,0).astype('uint8')
        print md.calculateLineMiddle()
        middle = md.calculateLineMiddle()
        cv2.circle(edges2,middle,1000,255,3)
        
        mark.drawHoughLines([md.mirror_line_Hough],edges2)
        mark.drawHoughLines([(md.mirror_line_Hough[0]+30,md.mirror_line_Hough[1])],edges2) 
        mark.drawHoughLines([(md.mirror_line_Hough[0]-30,md.mirror_line_Hough[1])],edges2)  
        
        print edges2.shape
        print len(np.nonzero(edges)[0])
        print len(np.nonzero(edges2)[0])
        filename = '../img/results/test.JPG'
        cv2.circle(direct.mask,middle,1000,0,-1)
        direct.getFiltredImge()
        cv2.imwrite(filename,direct.image)
        
            
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
    