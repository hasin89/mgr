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
from scene import edgeDetector

from skimage import morphology
from skimage import measure

import func.analise as an
from math import sqrt

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
    
    def openOperation(self,res,label,kernelSize = 3):
        background = np.where(res == label ,255,0).astype('uint8')
        kernel = np.ones((kernelSize,kernelSize),np.uint8)
        background = cv2.dilate(background,kernel,iterations = 1)
        background = cv2.erode(background,kernel,iterations = 1)
        res = np.where(background == 255,label,res)
    
        return res
    
    def getContour(self,points):
        contour = []
        stack = map(tuple,list(points))
        
        start = stack[0]
        p1 = start
        y,x = p1
        positions = [ 
            (y-1,x-1),
            (y-1,x),
            (y-1,x+1),
            (y,x-1),
            (y,x+1),
            (y+1,x-1),
            (y+1,x),
            (y+1,x+1)
            ]
        begining = []
        for p in positions:
            if p in stack:
                i = stack.index(p)
                p2 = stack.pop(i)
                begining.append(p2)
        if len(begining) == 2:
            contour.insert(len(contour),begining[0])
            contour.insert(0,begining[1])
        elif len(begining) ==1:
            contour.append(begining[0])
            begining.append(None)
        
        p1 = begining[0]
        while len(stack)>0:
            y,x = p1
            positions = [ 
                (y-1,x-1),
                (y-1,x),
                (y-1,x+1),
                (y,x-1),
                (y,x+1),
                (y+1,x-1),
                (y+1,x),
                (y+1,x+1)
                ]
            counter = 0
            for p in positions:
                if p in stack:
                    i = stack.index(p)
                    p2 = stack.pop(i)
                    contour.append(p2)
                    p1 = p2
                else:
                    counter = counter+1
            if counter == 8:
                break
            
        if begining[1] is not None:
            print 'not none'
            p1 = begining[1]
            while len(stack)>0:
                y,x = p1
                positions = [ 
                    (y-1,x-1),
                    (y-1,x),
                    (y-1,x+1),
                    (y,x-1),
                    (y,x+1),
                    (y+1,x-1),
                    (y+1,x),
                    (y+1,x+1)
                    ]
                counter = 0
                for p in positions:
                    if p in stack:
                        i = stack.index(p)
                        p2 = stack.pop(i)
                        contour.insert(0,p2)
                        p1 = p2
                    else:
                        counter = counter+1
                if counter == 8:
                    break
            
        print contour
        return contour
            
    
    def findCorner(self,contour,size):
        if len(contour)>size:
            points = []
            
            
            return points
        else:
            return []
        
    
    def findCornersOnContour(self,contour,size):
        '''
            size - dlugosc probierza, offset pomiedzy elementami konturu dla ktorych zrobiony jest odcinek
        '''
    
        if len(contour)>size:
            indexes = []
            dist = an.calcDistances(contour,size,int(size*0.1))
    
            for d in dist.iterkeys():
                segment1 = dist[d]
                MaxValue = max(np.asarray(segment1)[:,1])
                index = np.where(segment1 == MaxValue)[0]
                indexes.append(segment1[index[0]][0])
            return indexes
        else:
            return []
    
    def countFieldOfLabel(self,res,treshold=50,add_to_biggest=True):
        uni =  np.unique(res)
        
        counts = []
        for k,j in enumerate(uni):
            if j == -1:
                continue
            count = np.where(res == j,1,0)
            a = np.count_nonzero(count)
            # ignore meaningless labels
            if a < treshold:
                # save these points as background color
                res = np.where(res == j,-2,res)
                a = 0
            counts.append(a)
        
        labelsI = np.nonzero(counts)[0]
        
        # find the background label
        maxI = np.argmax(counts)
        
#         print counts
#         print counts[labelsI[0]]
#         print 'labels', labelsI
#         print max(counts)
#         print 'max', maxI
        
        # save these points as background color (now there is the background label number known)
        if add_to_biggest:
            res = np.where(res == -2,maxI,res)
        else:
            res = np.where(res == -2,-1,res)
        return res, labelsI, maxI
    
    def colorLabels(self,image,res,background_label = -1):
        #kolorowanie etykiet
        uni =  np.unique(res)
        colors = getColors(len(uni))
        for k,j in enumerate(uni):
            if j == background_label:
                continue
            color = np.asarray(colors[k])
            image[res == j] = color
        
        return image
    
    def fragmentation(self,skeleton):
        fragments = []
        points = np.nonzero(skeleton)
        points = np.transpose(points)
        nodes = []
        for p in points:
            y = p[0]
            x = p[1]
            mask = skeleton[y-1:y+2,x-1:x+2]
            ones = np.nonzero(mask)
            neibours = ones[0].size
            if neibours>3:
                nodes.append((y,x))
            pass
        skeleton2 = skeleton.copy()
        for node in nodes:
            skeleton2[node] = 0
        
        resT = measure.label(skeleton2,neighbors=8,background=0)
        resT = np.asarray(resT)
        resT.reshape(resT.shape[0],resT.shape[1],1)
        
        uni =  np.unique(resT)
        print 'unique l',uni
        
        resT,labelsIT,maxIT = self.countFieldOfLabel(resT,11,False)
        
        uni =  np.unique(resT)
        print 'unique l',uni
        
        return skeleton2, resT ,labelsIT,nodes
    
    def getWallContour(self,area_dists,labelsT,resT,img,Bmap):
        contours = {}
        #szukanie konturow nalezacych do scian
        for wall_label,dist_map in area_dists.iteritems():
            edgesMap2 = Bmap.copy()
            edgesMap2[:] = 0
            areaIMG = np.where(dist_map<1,1,0)
            
            image6 = img.copy()
            #caly bialy
            image6[areaIMG > -1] = (0,0,0)
#             image6[areaIMG == 1] = (0,255,255)
            
            contours[wall_label] = []
            for edge_label in labelsT:
                
                indexes = np.where(resT == edge_label)
                values = dist_map[indexes]
                
                max_value = np.max(values)
                min_value = np.max(values)
                if max_value < 20 and min_value>1:
                    
                    contour = np.transpose(indexes)
                    
                    contour = self.getContour(contour)
                    contours[wall_label].append(contour)
                    
                    image6[indexes] = (255,255,0)
                    
                    edgesMap2[indexes] = 1
                    
                    
                    
        return image6,edgesMap2,contours
    
    def getSobel(self,img,folder, i, letter):
        pic = i
        gauss_kernel = 5
        img = cv2.GaussianBlur(img, (gauss_kernel, gauss_kernel), 0)
        ed = edgeDetector.edgeDetector(img)
        
        r0 = ed.SobelChanel('R')
        g0 = ed.SobelChanel('G')
        b0 = ed.SobelChanel('B')
        
        fin = cv2.add(b0,g0)
        fin = cv2.add(fin,r0)
        
        image = img.copy()
        #caly bialy
        image[fin > -1] = (0,0,0)
        
        mask = np.where(fin>0,1,0).astype('uint8')
        
        CNTmask = mask.copy()
        
        distanceTreshold = 1
#         dst = cv2.distanceTransform(mask,cv2.cv.CV_DIST_C,3)
#         mask = np.where(dst>distanceTreshold,1,0).astype('uint8')
        
        
        kernel = np.ones((3,3),np.uint8)
        
#         image = cv2.erode(image,kernel,iterations = 1)
#         image = cv2.dilate(image,kernel,iterations = 1)
        
#         mask = cv2.erode(mask,kernel,iterations = 1)
#         mask = cv2.dilate(mask,kernel,iterations = 1)
        
        image2 = image.copy()
        image3 = image.copy()
        image[mask == 1] = (255,255,255)
                
        res = measure.label(mask,neighbors=8,background=1)
        res = np.asarray(res)
        res.reshape(res.shape[0],res.shape[1],1)
        
        res,labelsI,maxI = self.countFieldOfLabel(res)
        
        label = maxI
        res = self.openOperation(res, label, kernelSize=5)
        
        print 'unique', np.unique(res)
        
        # nie wiadomo czy to potrzebne, bo w sumie szeciany maja w miare jednolite pola
        defs = []
        cnts = []
        hulls = []
        area_dists = {}
        for label in labelsI:
            if label == maxI:
                continue
            res = self.openOperation(res, label, kernelSize=9)
            
            areaIMG = image.copy()
            areaIMG[:] = (0,0,0)
            area = np.where(res == label ,1,0).astype('uint8')
            
            area2 = np.where(res == label ,0,1).astype('uint8')
            area_dists[label] = cv2.distanceTransform(area2,cv2.cv.CV_DIST_L1,3)
            
            contours = cv2.findContours(area,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            cnt = contours[0][0]
            
            hull = cv2.convexHull(cnt,returnPoints = False)
            hulls.append(hull)
            defects = cv2.convexityDefects(cnt,hull)
            defs.append(defects)
            cnts.append(cnt)
            
            
#             pass
        
#             f = '../img/results/automated/%d/objects2/processed/%d_objects_on_mirror_%s_areaIMG.jpg' % (folder, i,letter)
#             print 'savaing to ' + f
#             cv2.imwrite(f, areaIMG)
        
        # szkieletyzacja
        mask[res == maxI] = 0
        edgesMap = np.where(res == -1,1,0).astype('uint8') 
#         edgesMap = mask.copy()
        mask2 = morphology.skeletonize(edgesMap > 0)
        mask2 = mask2.astype('uint8') 
        
        
        mask2,resT,labelsT,nodes = self.fragmentation(mask2)
        
        image6,edgesMap2,contours = self.getWallContour(area_dists, labelsT, resT, img, CNTmask)
        
                    
        for wall_label,dist_map in area_dists.iteritems():
                    
#                     cv2.circle(image6,(contour[0][1],contour[0][0]),2,(100,15,255),-1)
#                     cv2.circle(image6,(contour[-1][1],contour[-1][0]),2,(100,15,255),-1)
                    
#                     points = self.findCornersOnContour(contour, 100)
#                     for p in points:
#                         pass
#                         cv2.circle(image6,(contour[p][1],contour[p][0]),5,(255,0,255),2)
            
                    
                    
            for node in nodes:
                value = dist_map[node]
                if value < 20:
                    #node belong to the wall
                    pass
#                     image6[node] = (255,255,0)
            #znaldz linie hougha
            rho = 2
            theta = np.pi/90
            threshold = 20
                    
            lines = cv2.HoughLines(edgesMap2,rho,theta,threshold)
#             if lines is not None:
#                 for line in lines:
#                     line = line[0]
#                     cv2.line(image6, (line[0],line[1]), (line[2],line[3]), (128,0,128), 1)
            
            if lines is not None and len(lines[0])>0:
#                 mark.drawHoughLines(lines[0][:8], image6, (128,0,128), 1)
                pass
            
            image6 = self.interpolate(image6, contours[wall_label])
            
            edgesMap3 = CNTmask.copy()
            edgesMap3[:] = 0
#             edgesMap3[image6 == (255,255,0)] = 1
            
            
            f = '../img/results/automated/%d/objects2/linie/%d_objects_on_mirror_%s_%s_areaIMG.jpg' % (folder, i,letter,wall_label)
            print 'savaing to ' + f
            cv2.imwrite(f, image6)
            pass
        
        skeleton = mask2.copy()
        
        image5 = image2.copy()
        image5 = self.colorLabels(image5, resT, -1)
        
        f = '../img/results/automated/%d/objects2/linie/%d_objects_on_mirror_%s_skeleton.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, image5)
        
        image2[skeleton == 1] = (255,255,255)

        cnt,hi = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
        #znalezienie lini na szkielecie
        
        image3[skeleton == 1] = (255,255,255)
                
        rho = 0.5
        theta = np.pi/45
        part = 25
        
        rho = 1
        theta = np.pi/45
        part = 10
        
        threshold=int(skeleton.shape[1]/part)
#         threshold = 10
        
        #znaldz linie hougha
        lines = cv2.HoughLines(skeleton,rho,theta,threshold)
        if len(lines[0])>0:
            mark.drawHoughLines(lines[0], img, (128,0,128), 1)
            
            img[skeleton == 1] = (255,0,0)
        
#         image3[maps == 1] = (255,0,255)
        
        f = '../img/results/automated/%d/objects2/linie/%d_objects_on_mirror_%s_linie.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, img)
        
#         print 'contours', cont
#         print 'hierarhy', hi

        cont = {}
        print len(cnt)
        for k in range(len(cnt)):
            cont[k] = [map(tuple,j)[0] for j in cnt[k]]
            
        
            
        image2 = self.colorLabels(image2,res)
            
        
        
        kk = 0    
        for kk, defects in enumerate(defs): 
            image3 = image2.copy()
            cnt = cnts[kk]
            for jj in range(defects.shape[0]):
                s,e,f,d = defects[jj,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                line = an.getLine(start, end,0)
                distance = an.calcDistFromLine(far, line)
                
                cv2.line(image3,start,end,[255,255,255],2)
                if distance > 20:
                    print 'wielobok wypukly'
                    cv2.circle(image3,far,5,[0,0,255],-1)
            
            f = '../img/results/automated/%d/objects2/processed/%d_objects_on_mirror_%s_skeleton_%d.jpg' % (folder, pic,letter,kk)
            print 'savaing to ' + f
            cv2.imwrite(f, image3)
            
        for kk,c in enumerate(cnts):
            image4 = image2.copy()
            points = []
            for x in c:
                point = tuple(x[0])
                points.append(point)
            indexes = self.findCornersOnContour(points, 50)
            print 'indexes', indexes
            for ii in indexes:
                cv2.circle(image4,points[ii],5,(255,0,255),2)
            for p in self.POINTS:
                
                cv2.circle(image4,p,10,(255,255,0),3)
            
            f = '../img/results/automated/%d/objects2/processed/%d_objects_on_mirror_%s_hull_%d.jpg' % (folder, pic,letter,kk)
            print 'savaing to ' + f
            cv2.imwrite(f, image4)
            
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
        
        imgA,maskA = self.getSobel(zoneA.view,folder, i,'A')
        imgB,maskB = self.getSobel(zoneB.view,folder, i,'B')
        imgC,maskC = self.getSobel(zoneC.view,folder, i,'C')
        imgD,maskD = self.getSobel(zoneD.view,folder, i,'D')
        
        
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
        
        
        f = '../img/results/automated/%d/objects2/processed/%d_objects_on_mirror_%s.jpg' % (folder, i,'A')
        print 'savaing to ' + f
        cv2.imwrite(f, imgA)
        
        f = '../img/results/automated/%d/objects2/processed/%d_objects_on_mirror_%s.jpg' % (folder, i,'B')
        print 'savaing to ' + f
        cv2.imwrite(f, imgB)
        
        f = '../img/results/automated/%d/objects2/processed/%d_objects_on_mirror_%s.jpg' % (folder, i,'C')
        print 'savaing to ' + f
        cv2.imwrite(f, imgC)
        
        f = '../img/results/automated/%d/objects2/processed/%d_objects_on_mirror_%s.jpg' % (folder, i,'D')
        print 'savaing to ' + f
        cv2.imwrite(f, imgD)
        
    
#     def test_9_8(self):
#         self.execute(9, 8)
        
    def test_9_1(self):
        self.execute(9, 1)
        
    def test_9_2(self):
        self.execute(9, 2)
#         
    def test_9_3(self):
        self.execute(9, 3)
         
    def test_9_7(self):
        self.execute(9, 7)
         
         
    def test_9_10(self):
        self.execute(9, 10)
     
    def test_9_11(self):
        self.execute(9, 11)           

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
    
