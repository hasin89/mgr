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
from drawings.Draw import getColors

from calculations.labeling import LabelFactory
from drawings.Draw import getColors

from scene import analyticGeometry

class SobelTest(unittest.TestCase):


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
    
    def findObject(self,origin,folder,pic,letter):
        lf = LabelFactory(origin)
        e1 = cv2.getTickCount()
        lf.run(origin)
        e2 = cv2.getTickCount()
        print "time "+str( (e2-e1)/cv2.getTickFrequency())

        lf.flattenLabels()
        
        print 'preview:'
        e1 = cv2.getTickCount()
#         colorSpace = lf.getPreview()
        e2 = cv2.getTickCount()
        print "time "+str( (e2-e1)/cv2.getTickFrequency())
        
        contours = lf.convert2ContoursForm()
        
        cd = ContourDetector(origin)
        
        print 'find objects:'
        e1 = cv2.getTickCount()
        objects,margin = cd.findObjects(contours)
        print 'margin ',margin
        e2 = cv2.getTickCount()
        print "time "+str( (e2-e1)/cv2.getTickFrequency())
        
        center = []
        smallSquares = []
        area = []
        rects = []
        small = []
        circles = []
        rectangles = []
        
        for j,BND in enumerate(objects):
            x,y,w,h = cv2.boundingRect(BND)
            
            #filtracja obiektow po prawej i lewej krawedzi oraz z dolu
            if x == 1 or x+w+1 == origin.shape[1] or y+h+1 == origin.shape[0]:
                
                center.append(None)
                area.append(None)
                rects.append(None)
                rectangles.append(None)
                
                continue
            
#             if y == 1 and letter in ['C','D']:
#                 continue
            center_i = (h/2,w/2)
            area_i = h*w
            
            center.append(center_i)
            area.append(area_i)
            rects.append((x,y,w,h))
            
            if abs(h-w) < max(h,w)*0.1:
                #przypadek kwadratu
                
                
                if abs ( margin*2 - w ) < w*0.1:
                    print 'small square: ',(x,y, h, w)
                    small.append(j)
                    test1 = np.zeros_like(origin)
                    cv2.circle(test1,(int(x+w/2),int(y+h/2)),int(margin*2.5),200,1)
                    cv2.circle(origin,(int(x+w/2),int(y+h/2)),int(margin*2.5),200,1)
                    c1 = np.nonzero(test1)
                    c11 = np.transpose(c1)
                    circles.append(map(tuple,c11))
                    
                    
                #cv2.circle(origin,(int(x+w/2),int(y+h/2)),margin*2.5,200,2)
            test2 = np.zeros_like(origin)
            cv2.rectangle(test2,(x,y),(x+w,y+h),(255),1)
            cv2.rectangle(origin,(x,y),(x+w,y+h),(255),1)
            c2 = np.nonzero(test2)
            c22 = np.transpose(c2)
            rectangles.append(map(tuple,c22))
            
            
            
            
#             mark object
#             for i in range(0,len(BND)-1):
#                 cv2.line(origin,(BND[i][0][0],BND[i][0][1]) ,(BND[i+1][0][0],BND[i+1][0][1]),255,1)
                
        iMax = area.index(max(area))
        
        common = []
        if len(circles) > 0:
            for c in circles:
                print 'common points'
                common_points = set(rectangles[iMax]).intersection(c)
                if len(common_points)>0:
                    print 'YES'
                    common.append(c)
                    
                else:
                    print 'NO'
        points = rectangles[iMax]            
        for c in common:
            points = points + c
        
        BND2 = np.asarray([[(y,x) for (x,y) in points]])
#         print BND2
        x,y,w,h = cv2.boundingRect(BND2)
        cv2.rectangle(origin,(x,y),(x+w,y+h),(255),3) 
        
        padding = int ( w*0.1 )
        print 'width:', padding
        
        f = '../img/results/automated/%d/objects/%d_objects_on_mirror_A2_color_%s.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, origin)
        
        return (x ,y-padding,w,h)
                                   
#             for j in small:
#                 x,y,w,h = rects[j]
#                 xx,yy,ww,hh = rects[iMax]
#                 if (
#                     xx < x < xx + ww and (
#                                           abs(y - yy) < margin * 2.5 or abs(y - (yy+hh)) < margin * 2.5
#                                           ) 
#                     ) or ( 
#                           yy < y < yy + hh and (
#                                                 abs(x - xx) < margin * 2.5 or abs(x - (xx+ww)) < margin * 2.5
#                                                 )
#                           ):
#                     appendix.append(j)
                    
        
#         for i in appendix:
#             x,y,w,h = rects[i]
#             cv2.circle(origin,(int(x+w/2),int(y+h/2)),int(margin*2.5),200,1)
#             print 'small:', rects[i]
            
#         print 'big', rects[iMax]
#         BND = objects[iMax]
#         for i in range(0,len(BND)-1):
#             mark.drawMain(origin,(BND[i][0][0],BND[i][0][1]) ,(BND[i+1][0][0],BND[i+1][0][1]))
        
        f = '../img/results/automated/%d/objects/%d_objects_on_mirror_A_color_%s.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
#         cv2.imwrite(f, colorSpace)
                   
        f = '../img/results/automated/%d/objects/%d_objects_on_mirror_A2_color_%s.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, origin)
    
    
    def combine(self,res2,res3):
        res2 = np.asarray(res2,dtype='int64')
        res3 = np.asarray(res3,dtype='int64')
        
        r22 = pow(res2,2)
        r33 = pow(res3,2)
        
        rr = r22+r33
        r = np.zeros_like(r22)
        np.sqrt(rr,r)
        #copied from GIMP
#         r = r*2
        return r
    
    def SobelChanel(self,scene,chanel):
        res2 = cv2.Sobel(scene.view[:,:,chanel],-1,0,1,ksize=3,delta=0)
        res3 = cv2.Sobel(scene.view[:,:,chanel],-1,1,0,ksize=3,delta=0)
        print 2
        fl = cv2.flip(scene.view,-1)
        fl2 = cv2.Sobel(fl[:,:,chanel],-1,0,1,ksize=3,delta=0)
        fl3 = cv2.Sobel(fl[:,:,chanel],-1,1,0,ksize=3,delta=0)
        print 2
        r = self.combine(res2, res3)
        fl0 = self.combine(fl2, fl3)
        print 2
        fl_org = cv2.flip(fl0,-1)
#         r0 = cv2.addWeighted(r,1.0,fl_org,1.0,gamma=0)
        r0 = cv2.add(r,fl_org)
        
        r0 = np.where(r0<25,0,r0)
        
        return r0
    
    
    def findMirror(self, folder, pic):
        i = pic
        factor = 1
        
        filename = '../img/%d/%d.JPG' % (folder, i)
        
        scene = self.loadImage(filename, factor)
        
        gauss_kernel = 5
        gray_filtred = cv2.GaussianBlur(scene.gray, (gauss_kernel, gauss_kernel), 0)
        
        
        thrs1 = 20
        kernel = 9
        ratio = 3
        edge_filtred = cv2.Canny(gray_filtred, thrs1, thrs1 * ratio, kernel)
        
        res = cv2.Laplacian(scene.view,cv2.CV_64F)
        print 1
        r0 = self.SobelChanel(scene, 2)
        print 1
        g0 = self.SobelChanel(scene, 1)
        print 1
        b0 = self.SobelChanel(scene, 0)
        print 1
#         fin = cv2.addWeighted(r0 ,1.0,g0,1.0,gamma=0)
#         fin = cv2.addWeighted(fin,1.0,r0,1.0,gamma=0)
        fin = cv2.add(b0,g0)
        fin = cv2.add(fin,r0)
        r0 = fin
        
        r0 = np.where(r0>255,255,r0)
#         r0 = np.where(r0<75,0,r0)
        
        f = '../img/results/automated/%d/%d_laplasian.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, r0)
        
        
        f = '../img/results/automated/%d/%d_gray.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, gray_filtred)
        
        
        
        f = '../img/results/automated/%d/%d_canny.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, edge_filtred)
        
            
    def test_9_8(self):
        self.findMirror(9, 8)
        
            
            
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
    
