# -*- coding: utf-8 -*-
'''
Created on Nov 11, 2014

@author: Tomasz
'''
import unittest
import numpy as np
import cv2
from calculations.labeling import LabelFactory
from scene.mirrorDetector import mirrorDetector

class LocalizationTest(unittest.TestCase):


    def setUp(self):
        np.set_printoptions(precision=4)


    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def label(self,folder,pic):
        i = pic
        factor = 1
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        scene = self.loadImage(filename, factor)
        
        md = mirrorDetector(scene)
        try:
            mirror_zone = md.findMirrorZone()
        except Exception as e:
            raise e
        k = 25
        kernel = np.ones((k,k))
        dilated = cv2.dilate(md.edges_mask,kernel)
        edge = np.where(dilated>0,1,0)
        
        lf = LabelFactory(edge)
        
#         try:
        e1 = cv2.getTickCount()
        lf.run(edge)
        e2 = cv2.getTickCount()
        print "time "+str( (e2-e1)/cv2.getTickFrequency())
        print lf.P
        
#         lf2 = LabelFactory(edge)
        
#         time2 = self.l2(lf2)
#         print "time "+str(time2/cv2.getTickFrequency())
#         print lf2.P
        
#         except Exception as e:
        output = np.where(lf.L>1,255,0)
        f = '../img/results/automated/%d/%d_labeling.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, output)
#             raise e
            
    def l2(self,factory):
        
        e1 = cv2.getTickCount()
        factory.run2()
            
        e2 = cv2.getTickCount()
        time = e2-e1    
        return time
        
    
    def test_8_3(self):
        self.label(8, 3)
        
    def tes1t_8_6(self):
        self.label(8, 6)   
        
    def tes1t_8_7(self):
        self.label(8, 7) 
    
    def tes1t_8_19(self):
        self.label(8, 19) 
        
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