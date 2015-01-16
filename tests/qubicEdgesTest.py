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

class QubicEdgesTest(unittest.TestCase):
    
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
    
    def interpolate(self,image6,contours):
        '''
            contours[wall_label]
        '''
        #polaczenie koncow konturow tej samej sciany
        points = []
        tresh = 15
        for c in contours:
            for d in contours:
                if d == c:
                    continue
                dy = abs ( c[0][0] - d[0][0] )
                dx = abs ( c[0][1] - d[0][1] )
                
                dist = sqrt( dx*dx + dy*dy )
                
                if dist < tresh:
                    cv2.line(image6,(c[0][1],c[0][0]),(d[0][1],d[0][0]),(255,0,255), 1)
                    
                dy = abs ( c[0][0] - d[-1][0] )
                dx = abs ( c[0][1] - d[-1][1] )
                
                dist = sqrt( dx*dx + dy*dy )
                
                if dist < tresh:
                    cv2.line(image6,(c[0][1],c[0][0]),(d[-1][1],d[-1][0]),(255,0,255), 1)
                    
                dy = abs ( c[-1][0] - d[0][0] )
                dx = abs ( c[-1][1] - d[0][1] )
                
                dist = sqrt( dx*dx + dy*dy )
                
                if dist < tresh:
                    cv2.line(image6,(c[-1][1],c[-1][0]),(d[0][1],d[0][0]),(255,0,255), 1)
                    
                dy = abs ( c[-1][0] - d[-1][0] )
                dx = abs ( c[-1][1] - d[-1][1] )
                
                dist = sqrt( dx*dx + dy*dy )
                
                if dist < tresh:
                    cv2.line(image6,(c[-1][1],c[-1][0]),(d[-1][1],d[-1][0]),(255,0,255), 1)
        return image6
     
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
        image2 = image.copy()
        
        image3 = image.copy()
        
        
        
        image5 = image2.copy()
        lf = LabelFactory()
        image5 = lf.colorLabels(image5, qubic.edgeLabelsMap, -1)
        
        f = '../img/results/automated/%d/obj3/%d_objects_on_mirror_%s_skeleton.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, image5)
        
        image1 = image2.copy()
        image1[qubic.edgeMask == 1] = (255,255,255)
        f = '../img/results/automated/%d/obj3/%d_objects_on_mirror_%s_sobel.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, image1)
        
        image2[qubic.skeleton2 == 1] = (255,255,255)

        
        #znalezienie lini na szkielecie
        rho = 0.5
        theta = np.pi/45
        part = 25
        
        rho = 1
        theta = np.pi/45
        part = 10
        
            
        image2 = lf.colorLabels(image2,qubic.labelsMap)
            
        
        for kk,wall in qubic.walls.iteritems():
            image4 = image2.copy()
            corners = wall.findPotentialCorners()
            for p in corners:
                cv2.circle(image4,p,5,(255,0,255),2)
                
            for p in self.POINTS:
                cv2.circle(image4,p,10,(255,255,0),3)
            
            f = '../img/results/automated/%d/obj3/%d_objects_on_mirror_%s_hull_%d.jpg' % (folder, pic,letter,kk)
            print 'savaing to ' + f
            cv2.imwrite(f, image4)
            
            image3 = image.copy()
            for c in wall.contours:
                
                polygon  = c.polygon
                if len(polygon)>1:
                    for i in range(len(polygon)-1):
                        cv2.line(image3, (polygon[i][0][1],polygon[i][0][0]), (polygon[i+1][0][1],polygon[i+1][0][0]),(255,255,255),1)
                
                ll = map(np.array,np.transpose(np.array(c.points)))
                image3[ll] = (0,255,255)
                c.getLines()
                lines = c.lines
                if lines is not None:
#                     print lines
                    if c.wayPoint is not None:
                        mark.drawHoughLines(lines, image3, (128,0,128), 1)
                    elif len(polygon)>2:
                        mark.drawHoughLines(lines[:len(polygon)-1], image3, (128,0,128), 1)
                    else:
                        mark.drawHoughLines(lines[:1], image3, (128,0,128), 1)
            for node in wall.nodes:
#                 ll = map(np.array,np.transpose(np.array(c.points)))
                cv2.circle(image3,(node[1],node[0]),2,(255,255,0),-1)
                
            
            
            f = '../img/results/automated/%d/obj3/%d_common_edge_%s_%d.jpg' % (folder, pic,letter,kk)
            print 'savaing to ' + f
            cv2.imwrite(f, image3)
            
        return image,mask
    
    def getGrabCut(self,image,mask1,objectPoint):
        
        h = image.shape[0]
        w = image.shape[1]
        
        rect = (1, 1, w-1, h-1)
        print rect
        
        img = image.copy()
        mask = np.zeros(img.shape[:2],dtype = np.uint8)
        mask[ mask1 == 0] = 0
        
        #sure points
        pf = 1
        pb = 0
        
        cv2.circle(mask,(int(w/2) , int(h/2) ),3,pf,-1)
        
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        
        
        output = np.zeros(img.shape,np.uint8) 
           
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
        cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,10,cv2.GC_INIT_WITH_MASK)
        
        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask2)
        
        return output,mask
#         f = '../img/results/automated/%d/objects2/%d_objects_on_mirror_A.jpg' % (folder, i)
#         print 'savaing to ' + f
#         cv2.imwrite(f, output)
    
    def getBackProjection(self,img,mask):
        target = img.copy()
        
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        
        obj = np.nonzero(mask)
        y_tresh = obj[0].min()-20
        
        cv2.line(img, (0,y_tresh), (400,y_tresh), (255,0,0), 4)
        
        roi = Zone(img, 0 ,0, img.shape[1] , y_tresh)
        cv2.rectangle(img,(0,0),(img.shape[1] , y_tresh), (0,255,255),-1)
        roi = roi.image
        
        hsv_r = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        hsv_t = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
          
        # calculating object histogram
        roihist = cv2.calcHist([hsv_r],[0, 1], None, [180, 256], [0, 180, 0, 256] )
        
        # normalize histogram and apply backprojection
        cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
        dst = cv2.calcBackProject([hsv_t],[0,1],roihist,[0,180,0,256],1)
        
        # Now convolute with circular disc
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        cv2.filter2D(dst,-1,disc,dst)
        
        ret,thresh = cv2.threshold(dst,50,255,0)
        thresh = cv2.merge((thresh,thresh,thresh))
        res = cv2.bitwise_and(target,thresh)
        
        res = np.vstack((target,thresh,res))
#         
#         image = img.copy()
#         
#         image[fin > -1] = (0,0,0)
#         mask = np.where(fin>0,1,0).astype('uint8')

        y = obj[0].min()  
        index = np.where(obj[0]==y)
        print index[0][0]
        return res, mask, ( obj[index[0][0]] )

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
        
        
        mask = maskA
        rho = 1.5
        # theta = 0.025
        # rozdzielczosc kata
        theta = np.pi/182
        part = 3.5
        
        threshold=int(mask.shape[1]/part)
        threshold = 150
        
        #znaldz linie hougha
#         lines = cv2.HoughLines(mask,rho,theta,threshold)
#         if len(lines[0])>0:
#             mark.drawHoughLines(lines[0], image, (128,0,128), 1)
#             pass
#             self.scene.view[mask == 1] = (255,0,0)
        
        
        f = '../img/results/automated/%d/obj3/%d_objects_on_mirror_%s.jpg' % (folder, i,'A')
        print 'savaing to ' + f
        cv2.imwrite(f, imgA)
        
        f = '../img/results/automated/%d/obj3/%d_objects_on_mirror_%s.jpg' % (folder, i,'B')
        print 'savaing to ' + f
        cv2.imwrite(f, imgB)
        
        f = '../img/results/automated/%d/obj3/%d_objects_on_mirror_%s.jpg' % (folder, i,'C')
        print 'savaing to ' + f
        cv2.imwrite(f, imgC)
        
        f = '../img/results/automated/%d/obj3/%d_objects_on_mirror_%s.jpg' % (folder, i,'D')
        print 'savaing to ' + f
        cv2.imwrite(f, imgD)
        
    
#     def test_9_8(self):
#         self.execute(9, 8)
        
    def test_9_1(self):
        self.execute(9, 1)
        
#     def test_9_2(self):
#         self.execute(9, 2)
# #         
#     def test_9_3(self):
#         self.execute(9, 3)
#          
#     def test_9_7(self):
#         self.execute(9, 7)
# #          
# #          
#     def test_9_10(self):
#         self.execute(9, 10)
#       
#     def test_9_11(self):
#         self.execute(9, 11)           

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
    
