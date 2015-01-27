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
from numpy.core.numeric import dtype
import pickle
import numpy as np
from calculations import triangulation, DecompPMat, labeling
import cv2
import func.markElements as mark
from scene.mirrorDetector import mirrorDetector
from scene.zone import Zone
from scene.objectDetector2 import objectDetector2
from calculations.calibration import CalibrationFactory


class edgeDetectionTest(unittest.TestCase):


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
    
        
    def testCalibration(self):
        numBoards = 9#14
        board_w = 10
        board_h = 7
        flag = True
        
        
        CF = CalibrationFactory(numBoards,board_w,board_h,flag,'../img/10/','calibration/1/1/')
        CF.showCamerasPositions()
        
        mtx, dist, rvecs, tvecs = CF.mtx,CF.dist,CF.rvecs,CF.tvecs
            
        error = CF.reprojectPoints(CF.filenames)
        
        print 'errors',error
        
        
       
       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    