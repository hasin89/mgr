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

from scene.objectDetector2 import objectDetector2

from drawings.Draw import getColors

from calculations.labeling import LabelFactory
from drawings.Draw import getColors

from scene import analyticGeometry

class ObjectEdgesTest(unittest.TestCase):


    def setUp(self):
        np.set_printoptions(precision=4)


    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    

    def findMirror(self, folder, pic):
        i = pic
        factor = 1
        
        filename = '../img/results/automated/%d/objects2/%d_origin_A.jpg' % (folder,pic)
        
        scene = self.loadImage(filename, factor)
        
#         md = mirrorDetector(scene)
#             
#         mirror_zone = md.findMirrorZone()
#         
#         theta = 0
#         # middle X 
#         mid = int(mirror_zone.offsetX+mirror_zone.width/2)
#         mark.drawHoughLines([(mirror_zone.offsetX,theta),(mirror_zone.offsetX+mirror_zone.width,theta),(mid,theta)], scene.view, (255,0,0), 5)
#         
#         
# #         A|B
# #         ---
# #         C|D
#         od = objectDetector2(md,scene.view) 
#         zoneA,zoneB,zoneC,zoneD = od.detect()
        
        image = scene.view
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss_kernel = 5
        constant = 15
        blockSize = 101
        tresholdMaxValue = 255

#         gray_filtred = cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)
        
#         edge_filtred = cv2.adaptiveThreshold(gray_filtred,
#                                              maxValue=tresholdMaxValue,
#                                              adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                              thresholdType=cv2.THRESH_BINARY_INV,
#                                              blockSize=blockSize,
#                                              C=constant)
#         retv, edge_filtred = cv2.threshold(gray_filtred, 128, 255, cv2.THRESH_BINARY)
        
        
        x=0
        y=0
        
        
        print image.shape
        h = image.shape[0]
        w = image.shape[1]
        
        rect = (1, 1, w-1, h-1)
        print rect
        
        img = image.copy()
        mask = np.zeros(img.shape[:2],dtype = np.uint8)
        
        pf = 1
        pb = 0
        
        cv2.circle(mask,(0,0),3,pb,-1)
        cv2.circle(mask,(w-1,0),3,pb,-1)
        cv2.circle(mask,(w-1,h-1),3,pb,-1)
        
        cv2.circle(mask,(int(w/2) , int(h/2) ),3,pf,-1)
        
        

        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        
        
        output = np.zeros(img.shape,np.uint8) 
           
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,10,cv2.GC_INIT_WITH_MASK)
        
        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask2)
        
        pb = (0,0,255)
        pf = (0,255,0)
        
        cv2.circle(output,(0,0),3,pb,-1)
        cv2.circle(output,(w-1,0),3,pb,-1)
        cv2.circle(output,(w-1,h-1),3,pb,-1)
        
        cv2.circle(output,(int(w/2) , int(h/2) ),3,pf,-1)
        
        
        f = '../img/results/automated/%d/objects2/%d_objects_on_mirror_A.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, output)
        
        
#         f = '../img/results/automated/%d/objects2/%d_objects_on_mirror_B.jpg' % (folder, i)
#         print 'savaing to ' + f
#         cv2.imwrite(f, zoneB.image)
#         
#         f = '../img/results/automated/%d/objects2/%d_objects_on_mirror_C.jpg' % (folder, i)
#         print 'savaing to ' + f
#         cv2.imwrite(f, zoneC.image)
#         
#         f = '../img/results/automated/%d/objects2/%d_objects_on_mirror_D.jpg' % (folder, i)
#         print 'savaing to ' + f
#         cv2.imwrite(f, zoneD.image)
            

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
    
