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
            for i in range(0,len(BND)-1):
                cv2.line(origin,(BND[i][0][0],BND[i][0][1]) ,(BND[i+1][0][0],BND[i+1][0][1]),255,1)
                
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
        print rects          
        for c in common:
            points = points + c
        
        if points is not None:
            BND2 = np.asarray([[(y,x) for (x,y) in points]])
#         print BND2
            x,y,w,h = cv2.boundingRect(BND2)
            cv2.rectangle(origin,(x,y),(x+w,y+h),(255),5) 
        else:
            f = '../img/results/automated/%d/objects/%d_objects_on_mirror_A2_color_%s.jpg' % (folder, pic,letter)
            print 'savaing to ' + f
            cv2.imwrite(f, origin)
            return (0,0,0,0)
        
        padding = int ( w*0.1 )
        print 'width:', padding
        Y = y-padding
        
        #dodawanie oderwanych czesci duzych kwadratow
        maxi = rects[iMax]
        print 'first ROI:', area[iMax]
        if len(area)>1:
            area[iMax] = None
            iMax2 = area.index(max(area))
            
            if area[iMax2] != None:
                print 'second ROI:', area[iMax2]
                midi = rects[iMax2]
                
                if midi[0]>maxi[0] and midi[1]<maxi[1] and midi[2]<maxi[2]:
                    print 'Upper part'
                    print 'distance', abs(maxi[1] - midi[1]-midi[3] )
                    Y = midi[1]
                    print Y
                    h += midi[3] + abs(maxi[1] - midi[1]-midi[3] )
                print midi
                print maxi
            
        f = '../img/results/automated/%d/objects/%d_objects_on_mirror_A2_color_%s.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, origin)
        
        return (x ,Y,w,h)
                                   
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
        k = 127
        k = 1
        kernel = np.ones((k,k))
        dilated = cv2.dilate(md.edges_mask,kernel)
        mask = dilated
            
        edge = np.where(mask>0,255,0)
        
        f = '../img/results/automated/%d/%d_edge.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, edge)
        
        zoneA =  Zone(edge,   mirror_zone.offsetX ,mirror_zone.offsetY                                ,mid-mirror_zone.offsetX                                    ,md.calculatePointOnLine(mid)[1]-mirror_zone.offsetY)
        zoneB =  Zone(edge,   mid                 ,mirror_zone.offsetY                                ,mirror_zone.offsetX+mirror_zone.width-mid                  ,md.calculatePointOnLine(mirror_zone.offsetX+mirror_zone.width)[1]-mirror_zone.offsetY)
        zoneC =  Zone(edge,   mirror_zone.offsetX ,md.calculatePointOnLine(mirror_zone.offsetX)[1]    ,mid-mirror_zone.offsetX                                    ,mirror_zone.offsetY+mirror_zone.height - md.calculatePointOnLine(mirror_zone.offsetX)[1])
        zoneD =  Zone(edge,   mid                 ,md.calculatePointOnLine(mid)[1]                    ,mirror_zone.offsetX+mirror_zone.width-mid                  ,mirror_zone.offsetY+mirror_zone.height - md.calculatePointOnLine(mid)[1] )
        
        margin = 50
        zoneA = self.setMargin(zoneA, margin)
        zoneB = self.setMargin(zoneB, margin)
        zoneC = self.setMargin(zoneC, margin)
        zoneD = self.setMargin(zoneD, margin)
        
        origin = zoneA.image
        (x,y,w,h) = self.findObject(origin, folder, pic,'A')
        zoneA = Zone(scene.view,x+zoneA.offsetX,y+zoneA.offsetY,w,h)
        
        origin = zoneB.image
        (x,y,w,h) = self.findObject(origin, folder, pic,'B')
        zoneB = Zone(scene.view,x+zoneB.offsetX,y+zoneB.offsetY,w,h)
        
        origin = zoneC.image
        (x,y,w,h) = self.findObject(origin, folder, pic,'C')
        zoneC = Zone(scene.view,x+zoneC.offsetX,y+zoneC.offsetY,w,h)
        
        origin = zoneD.image
        (x,y,w,h) = self.findObject(origin, folder, pic,'D')
        zoneD = Zone(scene.view,x+zoneD.offsetX,y+zoneD.offsetY,w,h)
        
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
        
        f = '../img/results/automated/%d/objects2/%d_objects_on_mirror_A.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, zoneA.image)
        
        f = '../img/results/automated/%d/objects2/%d_objects_on_mirror_B.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, zoneB.image)
        
        f = '../img/results/automated/%d/objects2/%d_objects_on_mirror_C.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, zoneC.image)
        
        f = '../img/results/automated/%d/objects2/%d_objects_on_mirror_D.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, zoneD.image)
        
        f = '../img/results/automated/%d/%d_view.jpg' % (folder, i)
        print 'savaing to ' + f
        cv2.imwrite(f, scene.view)       

    def test_9_11(self):
        self.findMirror(9, 11)
         
    def test_9_10(self):
        self.findMirror(9, 10)   
        
    def test_9_8(self):
        self.findMirror(9, 8)
        
    def test_9_7(self):
        self.findMirror(9, 7)
        
    def test_9_6(self):
        self.findMirror(9, 6)
        
    def test_9_5(self):
        self.findMirror(9, 5)
        
    def test_9_4(self):
        self.findMirror(9, 4)    
        
    def test_9_3(self):
        self.findMirror(9, 3) 

    def test_9_2(self):
        self.findMirror(9, 2)
        
    def test_9_1(self):
        self.findMirror(9, 1)    
    
#     def test_8_19(self):
#         self.findMirror(8, 19) 
# 
#     def test_8_77(self):
#         self.findMirror(8, 77)

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
    
