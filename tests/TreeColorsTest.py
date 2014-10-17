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

class TreeColorsTest(unittest.TestCase):


    def setUp(self):
        np.set_printoptions(precision=4)


    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    
    def tes1tGrabCut(self):
        
        folder = 8
        i= 3
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        factor = 0.25
        scene = self.loadImage(filename, factor)
        
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        
        iterations = 2
        
        mask = np.zeros(scene.view.shape[:2],dtype = np.uint8)
        
        #draw mask point here
        cv2.circle(mask,(263,363),3,1,-1)
        cv2.circle(mask,(273,373),3,1,-1)
        cv2.circle(mask,(253,353),3,1,-1)
        
        
        x=263
        y=363
        
        
        width = 150
        height = 150
        rect = (x,y,width,height)
        img = scene.view.copy()
        output = np.zeros(img.shape,np.uint8)    
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,10,cv2.GC_INIT_WITH_MASK)
        
        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask2)
        filename = '../img/%d/%d.JPG' % (folder, i)
        f = '../img/results/matching/%d/folder_%d_%d_grabCUT_%d.jpg' % (folder,folder, i, factor)
        cv2.imwrite(f,output)
        
    def tes1tGrabCut2(self):
        
        folder = 8
        i= 19
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        factor = 1
        scene = self.loadImage(filename, factor)
        
        
        k = 5
        
        #scene.view = cv2.GaussianBlur(scene.view, (k, k), 0)
        
        #13*5
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        
        iterations = 2
        
        mask = np.zeros(scene.view.shape[:2],dtype = np.uint8)
        
        #draw mask point here
                
        cv2.circle(mask,(1100,1250),3,1,-1)
        cv2.circle(mask,(1111,1260),3,1,-1)
        cv2.circle(mask,(1000,1270),3,1,-1)
        
        
        x=1100
        y=1250
        
        width = 500
        height = 500
        rect = (x,y,width,height)
        
        img = scene.view.copy()
        
        cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),[255,0,0],2)
        
        f = '../img/results/matching/%d/folder_%d_%d_rectangle_%d.jpg' % (folder,folder, i, factor)
        cv2.imwrite(f,img)
        
        output = np.zeros(img.shape,np.uint8)    
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
        #cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,10,cv2.GC_INIT_WITH_MASK)
        
        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask2)
        filename = '../img/%d/%d.JPG' % (folder, i)
        f = '../img/results/matching/%d/folder_%d_%d_grabCUT_%d.jpg' % (folder,folder, i, factor)
        cv2.imwrite(f,output)
        
    def tes1tGrabCut3(self):
        
        folder = 8
        i= 3
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        factor = 1
        scene = self.loadImage(filename, factor)
        
        
        k = 3
        
        scene.view = cv2.GaussianBlur(scene.view, (k, k), 0)
        
        #13*5
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        
        iterations = 2
        
        mask = np.zeros(scene.view.shape[:2],dtype = np.uint8)
        
        #draw mask point here
        diamension = 5
        cv2.circle(mask,(1100,1250),diamension,1,-1)
        cv2.circle(mask,(1111,1260),diamension,1,-1)
        cv2.circle(mask,(1000,1270),diamension,1,-1)
        
        
        x=1150
        y=1500
        
        width = 500
        height = 600
        rect = (x,y,width,height)
        
        img = scene.view.copy()
        
        cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),[255,0,0],2)
        
        f = '../img/results/matching/%d/folder_%d_%d_rectangle_%d.jpg' % (folder,folder, i, factor)
        cv2.imwrite(f,img)
        
        output = np.zeros(img.shape,np.uint8)    
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
        
        
        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask2)
        filename = '../img/%d/%d.JPG' % (folder, i)
        f = '../img/results/matching/%d/folder_%d_%d_grabCUT_%d.jpg' % (folder,folder, i, factor)
        cv2.imwrite(f,output)
        
    def tes1tDistance1(self):
        folder = 8
        i= 77
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        scene = self.loadImage(filename)
        
        gamma = 0.45
        gauss_kernel = 11
        constant = 6
        blockSize = 51
        
        tresh = 25
        self.calcDist(scene,gauss_kernel,constant,blockSize,tresh,folder,i)
        
    def testDistance2(self):
        folder = 7
        i= 77
        i =2
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        factor = 1.000
        scene = self.loadImage(filename,factor)
        
        if factor == 1:
            gamma = 0.45
            gauss_kernel = 5
            constant = 5
            blockSize = 101
            tresh = 4
        else:
            
            gauss_kernel = 3
            constant = 3
            blockSize = 301
            tresh = 2
        
        
        self.calcDist(scene,gauss_kernel,constant,blockSize,tresh,folder,i)
        
    def testHorizontal(self):
        folder = 4
        i= 77
        i =10
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        print "horizotal"
        print filename
        factor = 1
        scene = self.loadImage(filename,factor)
        
        gauss_kernel = 5
        constant = 5
        blockSize = 101
        tresh = 4
        
        mask = self.calcDist(scene,gauss_kernel,constant,blockSize,tresh,folder,i)
        
        rho = 1.5
        # theta = 0.025
        theta = np.pi/180
        
        threshold=int(mask.shape[1]/3)
        
        #znaldz linie hougha
        
        lines2 = cv2.HoughLines(mask,rho,theta,threshold)
        
        Amin = 2
        for (rho,theta) in lines2[0][:2]:
            print (rho,theta)
            line = convertLineToGeneralForm((rho,theta),mask.shape)
            A = abs((round(line[0],0)))
            if A<Amin:
                mirror_line = line
                Amin=A 
        
        mirror_line
        
        vis1 = np.where(mask==1,255,0).astype('uint8')
        
        mark.drawHoughLines(lines2[0][:2],vis1) 
        
        
        reflected, direct,point = self.divide(mirror_line, vis1)
        print point
        cv2.circle(vis1, point ,150,255,-1)
        
        
        f = '../img/results/matching/%d/folder_%d_%d_test_Hough2_.jpg' % (folder,folder, i)
        cv2.imwrite(f,vis1)
        f = '../img/results/matching/%d/folder_%d_%d_test_dir_.jpg' % (folder,folder, i)
        cv2.imwrite(f,reflected)
        f = '../img/results/matching/%d/folder_%d_%d_test_ref_.jpg' % (folder,folder, i)
        cv2.imwrite(f,direct)
        
    def divide(self, mirror_line, img):
        
#         Ax+By+c self.width
        y = abs((mirror_line[0]*img.shape[0]/2.0+mirror_line[2])/mirror_line[1])
        reflected = img[:y, :]
        direct = img[y:, :]
        
        return reflected, direct,(int(img.shape[1]/2) , abs(int(y)))
        
        
    def calcDist(self,scene,gauss_kernel,constant,blockSize,tresh,folder,i):
        
        gray = scene.gray
        gray_filtred = cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)
        
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtred,
                                             maxValue=255,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=blockSize,
                                             C=constant)
                
        f = '../img/results/matching/%d/folder_%d_%d_test_distance_treshold_.jpg' % (folder,folder, i)
        cv2.imwrite(f,edge_filtred)
        
        dst = cv2.distanceTransform(edge_filtred,cv2.cv.CV_DIST_C,3)
        mask = np.where(dst>tresh,255,0).astype('uint8')
        f = '../img/results/matching/%d/folder_%d_%d_test_distance_map_.jpg' % (folder,folder, i)
        cv2.imwrite(f,mask)
        mask = np.where(dst>tresh,1,0).astype('uint8')
        return mask
    
    def tes1tForBlue(self):
        
        folder = 8
        i= 19
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        scene = self.loadImage(filename)
        gauss_kernel = 3
        constant = 3
        blockSize = 301
            
        gray = scene.view[:,:,0]
        gray_filtred = cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)
        
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtred,
                                             maxValue=255,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=blockSize,
                                             C=constant)
                
        f = '../img/results/matching/%d/folder_%d_%d_test_niebieski_.jpg' % (folder,folder, i)
        cv2.imwrite(f,edge_filtred)
        
    def tes1tForGreen(self):
        
        folder = 8
        i= 19
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        scene = self.loadImage(filename)
        gauss_kernel = 3
        constant = 3
        blockSize = 301
            
        gray = scene.view[:,:,1]
        gray_filtred = cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)
        
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtred,
                                             maxValue=255,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=blockSize,
                                             C=constant)
                
        f = '../img/results/matching/%d/folder_%d_%d_test_zielony_.jpg' % (folder,folder, i)
        cv2.imwrite(f,edge_filtred)
        
    def tes1tForRed(self):
        
        folder = 8
        i= 19
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        scene = self.loadImage(filename)
        gauss_kernel = 3
        constant = 3
        blockSize = 301
            
        gray = scene.view[:,:,2]
        gray_filtred = cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)
        
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtred,
                                             maxValue=255,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=blockSize,
                                             C=constant)
                
        f = '../img/results/matching/%d/folder_%d_%d_test_czerwony_.jpg' % (folder,folder, i)
        cv2.imwrite(f,edge_filtred)
        
    def tes1tForGray(self):
        
        folder = 8
        i= 19
                
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        scene = self.loadImage(filename)
        gauss_kernel = 3
        constant = 3
        blockSize = 301
            
        gray = scene.gray
        gray_filtred = cv2.GaussianBlur(gray, (gauss_kernel, gauss_kernel), 0)
        
        
        edge_filtred = cv2.adaptiveThreshold(gray_filtred,
                                             maxValue=255,
                                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             thresholdType=cv2.THRESH_BINARY_INV,
                                             blockSize=blockSize,
                                             C=constant)
                
        f = '../img/results/matching/%d/folder_%d_%d_test_szary_.jpg' % (folder,folder, i)
        cv2.imwrite(f,edge_filtred)

        
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
    