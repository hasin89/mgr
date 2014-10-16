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


class edgeDetectionTest(unittest.TestCase):


    def setUp(self):
        np.set_printoptions(precision=4)
        



    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def testTresholdPlusScalling(self):
        
        edge = None
        
        folder = 8
        i = 19
        i= 3
        i = 19
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        scene = self.loadImage(filename,0.25)
        scene.gamma = 0.45
        scene.gauss_kernel = 7
        scene.constant = 3
        scene.blockSize = 25
        
        edge = scene.getEdges()
                
        f = '../img/results/matching/%d/folder_%d_%d_edge_test_atresh_factor_025_.jpg' % (folder,folder, i)
        cv2.imwrite(f,edge.map)
        
    def testTresholdNoScalling(self):
        
        edge = None
        
        folder = 6
        i = 19
        i= 3
        i = 18
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        print filename
        scene = self.loadImage(filename,1)
        scene.gamma = 0.45
        scene.gauss_kernel = 13
        scene.constant = 6
        scene.blockSize = 51
        
        edge = scene.getEdges()
                
        f = '../img/results/matching/%d/folder_%d_%d_edge_test_atresh_factor_1_.jpg' % (folder,folder,i)
        cv2.imwrite(f,edge.map)
        
    def testCannyNoScalling(self):
        
        edge = None
        
        folder = 6
        i = 19
        i= 3
        i = 18
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        print filename
        scene = self.loadImage(filename,1)
        scene.gamma = 0.45
        scene.gauss_kernel = 3
        constant = 6
        blockSize = 51
        
        gray_filtred = cv2.GaussianBlur(scene.gray, (scene.gauss_kernel, scene.gauss_kernel), 0)

        
        thrs1 = 20
        kernel = 9
        ratio = 3
        edge_filtred = cv2.Canny(gray_filtred, thrs1, thrs1 * ratio, kernel)
                        
        f = '../img/results/matching/%d/folder_%d_%d_edge_test_canny_factor_1_.jpg' % (folder,folder,i)
        cv2.imwrite(f,edge_filtred)        
        
        
    def testtreshold(self):
        
        edge = None
        
        folder = 8
        i = 19
        i= 3
        i = 19
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        scene = self.loadImage(filename)
        scene.gamma = 0.45
        scene.gauss_kernel = 5
        scene.constant = 2
        scene.blockSize = 11
        
        edge = scene.getEdges()
                
        f = '../img/results/matching/%d/folder_%d_%d_edge_test22_.jpg' % (folder,folder, i)
        cv2.imwrite(f,edge.map)
        
        
    def testKwantyzacja(self):
        folder = 8
        i = 19
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        scene = self.loadImage(filename)
        
        gray = scene.gray
        
        gray2 = gray/16*16
        gray = gray2
                
        gamma = 1
        k = 11
        constant = 2
        blockSize = 11
        
        gray_filtred = cv2.GaussianBlur(gray, (k, k), 0)
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtred,
                                             maxValue=255,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=blockSize,
                                             C=constant)
        
        
        f = '../img/results/matching/%d/folder_%d_%d_gray_skalowanie.jpg' % (folder,folder, i)
        print f
        cv2.imwrite(f,edge_filtred)
        
        
        
        
    def tes1tRreshold(self):
        
        edge = None
        folder = 8
        i= 3
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        scene = self.loadImage(filename)
        
        gamma = 0.45
        gauss_kernel = 11
        constant = 2
        blockSize = 11
        
        
        gray_filtered = features.filterImg(scene.view, gauss_kernel, gamma)
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtered,
                                             maxValue=255,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=blockSize,
                                             C=constant)
        thrs1 = 20
        kernel = 11
        ratio = 3
#         edge_filtred = cv2.Canny(gray_filtered, thrs1, thrs1 * ratio, kernel)
    
        theta = 0.025 
        rho = 1
        threshold = 25
        
        theta = 0.025 
        rho = 0.5
        print np.pi/180
        theta = np.pi/45
        threshold = 100
        
        lines3 = cv2.HoughLinesP(edge_filtred,rho,theta,threshold,30,10)
        
        
        lines2 = cv2.HoughLines(edge_filtred,rho,theta,threshold)
        lines = lines2[0]
        
        vis_filtred = scene.view.copy()
        vis_filtred[edge_filtred != 0] = (0, 255, 0)
        vis_filtred[edge_filtred == 0] = (0, 0, 0)
#         
        mark.drawHoughLines(lines,vis_filtred) 
        
        for (x1,y1,x2,y2) in lines3[0]:
            pt1 = (x1,y1)
            pt2 = (x2,y2)
            cv2.line(vis_filtred, pt1, pt2, (128,0,128), 3)
        
#         lines2 = cv2.HoughLines(tmpbinary,rho,theta,threshold)
                
        f = '../img/results/matching/%d/folder_%d_%d_edge_test_2.jpg' % (folder,folder, i)
        cv2.imwrite(f,vis_filtred)
        
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
    