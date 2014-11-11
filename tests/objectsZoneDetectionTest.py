'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
from calculations import triangulation, DecompPMat
import cv2
import func.trackFeatures as features
from scene.edge import edgeMap
import func.markElements as mark
from scene.analyticGeometry import convertLineToGeneralForm
from scene.mirrorDetector import mirrorDetector
from scene.zone import Zone
from scene.ContourDectecting import ContourDetector
from scene.ObjectDectecting import ObjectDetector


class ObjectLocalizationTest(unittest.TestCase):


    def setUp(self):
        np.set_printoptions(precision=4)


    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    
    def setMargin(self,zone, margin):
        
        newOffsetX = zone.offsetX + margin 
        newOffsetY = zone.offsetY + margin
        newWidth = zone.width - margin - margin
        newHeight = zone.height - margin - margin
        
        zone = Zone(zone.origin, newOffsetX, newOffsetY, newWidth, newHeight)
        
        return zone 
    
    
    def findMirror(self, folder, pic):
        i = pic
        factor = 1
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        scene = self.loadImage(filename, factor)
        
        md = mirrorDetector(scene)
            
        try:
            mirror_zone = md.findMirrorZone()
        except Exception as e:
            f = '../img/results/automated/%d/%d_mirror_vertical.jpg' % (folder, i)
            print 'savaing to ' + f
            cv2.imwrite(f, md.scene.view)
            raise e
        
        theta = 0
        # middle X 
        mid = int(mirror_zone.offsetX+mirror_zone.width/2)
        mark.drawHoughLines([(mirror_zone.offsetX,theta),(mirror_zone.offsetX+mirror_zone.width,theta),(mid,theta)], scene.view, (255,0,0), 5)
        
        md.calculatePointOnLine(mid)
        
#         A|B
#         ---
#         C|D
        print '====='
        print scene.view.shape
        print mirror_zone.offsetX ,mirror_zone.offsetY                                ,mid                                    ,md.calculatePointOnLine(mid)[1]
        print '====='
        k = 25
        kernel = np.ones((k,k))
        dilated = cv2.dilate(md.edges_mask,kernel)
        edge = np.where(dilated>0,255,0)
        
        zoneA =  Zone(edge,   mirror_zone.offsetX ,mirror_zone.offsetY                                ,mid-mirror_zone.offsetX                                    ,md.calculatePointOnLine(mid)[1]-mirror_zone.offsetY)
        zoneB =  Zone(edge,   mid                 ,mirror_zone.offsetY                                ,mirror_zone.offsetX+mirror_zone.width-mid                  ,md.calculatePointOnLine(mirror_zone.offsetX+mirror_zone.width)[1]-mirror_zone.offsetY)
        zoneC =  Zone(edge,   mirror_zone.offsetX ,md.calculatePointOnLine(mirror_zone.offsetX)[1]    ,mid-mirror_zone.offsetX                                    ,mirror_zone.offsetY+mirror_zone.height - md.calculatePointOnLine(mirror_zone.offsetX)[1])
        zoneD =  Zone(edge,   mid                 ,md.calculatePointOnLine(mid)[1]                    ,mirror_zone.offsetX+mirror_zone.width-mid                  ,mirror_zone.offsetY+mirror_zone.height - md.calculatePointOnLine(mid)[1] )
        
        margin = 50
        zoneA = self.setMargin(zoneA, margin)
        zoneB = self.setMargin(zoneB, margin)
        zoneC = self.setMargin(zoneC, margin)
        zoneD = self.setMargin(zoneD, margin)
        
        cd = ContourDetector(zoneA.preview)
        contours = cd.findContours()
        objects = cd.findObjects(contours)
        
        for BND in objects:
            for i in range(0,len(BND)-1):
                mark.drawMain(zoneA.preview,(BND[i][0][0],BND[i][0][1]) ,(BND[i+1][0][0],BND[i+1][0][1]))
        
        
        
        
        f = '../img/results/automated/%d/%d_objects_on_mirror_A.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, zoneA.preview)
        
        f = '../img/results/automated/%d/%d_objects_on_mirror_B.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, zoneB.preview)
        
        f = '../img/results/automated/%d/%d_objects_on_mirror_C.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, zoneC.preview)
        
        f = '../img/results/automated/%d/%d_objects_on_mirror_D.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, zoneD.preview)
            

    def test_8_3(self):
        self.findMirror(8, 3)
        
    def test_8_6(self):
        self.findMirror(8, 6)   
        
    def test_8_7(self):
        self.findMirror(8, 7) 
    
    def test_8_19(self):
        self.findMirror(8, 19) 

    def test_8_77(self):
        self.findMirror(8, 77)

#     def test_7_1(self):
#         self.findMirror(7, 1)   
#         
#     def test_7_2(self):
#         self.findMirror(7, 2) 
#     
#     def test_7_3(self):
#         self.findMirror(7, 3) 
# 
#     def test_4_1(self):
#         self.findMirror(4, 1)   
#         
#     def test_4_2(self):
#         self.findMirror(4, 2) 
#     
#     def test_4_3(self):
#         self.findMirror(4, 3)
#     
#     def test_4_4(self):
#         self.findMirror(4, 4)   
#         
#     def test_4_5(self):
#         self.findMirror(4, 5) 
#     
#     def test_4_6(self):
#         self.findMirror(4, 6)
#     
#     def test_4_7(self):
#         self.findMirror(4, 7)   
#         
#     def test_4_8(self):
#         self.findMirror(4, 8) 
#     
#     def test_4_9(self):
#         self.findMirror(4, 9)  
# 
#     def test_5_1(self):
#         self.findMirror(5, 1)   
#         
#     def test_5_2(self):
#         self.findMirror(5, 2) 
#     
#     def test_5_3(self):
#         self.findMirror(5, 3)
#     
#     def test_5_4(self):
#         self.findMirror(5, 4) 
#         
#     def test_5_6(self):
#         self.findMirror(5, 6)
#     
#     def test_5_7(self):
#         self.findMirror(5, 7)   
#         
#     def test_5_8(self):
#         self.findMirror(5, 8) 
#     
#     def test_5_9(self):
#         self.findMirror(5, 9)
#         
#     def test_5_10(self):
#         self.findMirror(5, 10) 
#     
#     def test_5_11(self):
#         self.findMirror(5, 11)
            
            
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
    
