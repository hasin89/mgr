# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
from calculations import triangulation,DecompPMat
from scene.edge import edgeMap
import pickle
import cv2
import func.markElements as mark
from scene.mirrorDetector import mirrorDetector
from scene.zone import Zone
from scene.objectDetector2 import objectDetector2
from calculations.calibration import CalibrationFactory
from scene.scene import Scene
from scene import edgeDetector
from scene.qubic import QubicObject
import sys,os

from drawings.Draw import getColors

class edgeDetectionTest(unittest.TestCase):

    writepath = ''
    
    def setUp(self):
        np.set_printoptions(precision=4)

    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def getZones(self,folder,pic):
        path1 = self.writepath+ 'pickle_%d_zone_A.p' % (self.i)
        path2 = self.writepath+ 'pickle_%d_zone_C.p' % (self.i)
        path3 = self.writepath+ 'pickle_%d_scene.p' % (self.i)
        path4 = self.writepath+ 'pickle_%d_md.p' % (self.i)
        if os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3) and os.path.exists(path4):
            print 'reading'
            fname = path1
            f = open(fname,'rb')
            zoneA = pickle.load(f)
            f.close()
            fname = path2
            f = open(fname,'rb')
            zoneC = pickle.load(f)
            f.close()
            fname = path3
            f = open(fname,'rb')
            scene = pickle.load(f)
            f.close()
            fname = path4
            f = open(fname,'rb')
            md = pickle.load(f)
            f.close()
            
        else:
        
        
            factor = 1
            filename = '../img/%d/%d.JPG' % (folder, pic)
            scene = self.loadImage(filename, factor)
            
            md = mirrorDetector(scene)            
            
            img = np.where(md.edges_mask == 1,255,0)
            f = self.writepath+ '%d_edges.jpg' % (self.i)
            print 'savaing to ' + f
            cv2.imwrite(f, img)
            
            mz = md.findMirrorZone()
            
            f = self.writepath+ '%d_mirror_zone.jpg' % (self.i)
            print 'savaing to ' + f
            cv2.imwrite(f, mz.preview)
            
            f = self.writepath+ '%d_edges_detection.jpg' % (self.i)
            print 'savaing to ' + f
            cv2.imwrite(f, md.scene.view)
            
            od = objectDetector2(md,md.origin)
            zoneA,zoneC = od.detect(chessboard=True)
            
            f = self.writepath+ '%d_A_zone.jpg' % (self.i)
            print 'savaing to ' + f
            cv2.imwrite(f, zoneA.image)
            
            f = self.writepath+ '%d_C_zone.jpg' % (self.i)
            print 'savaing to ' + f
            cv2.imwrite(f, zoneC.image)
            
            #zapisz na póxniej
            fname = path1
            f = open(fname,'wb')
            pickle.dump(zoneA, f)
            f.close()
            
            fname = path2
            f = open(fname,'wb')
            pickle.dump(zoneC, f)
            f.close()
            
            fname = path3
            f = open(fname,'wb')
            pickle.dump(scene, f)
            f.close()
            
            fname = path4
            f = open(fname,'wb')
            pickle.dump(md, f)
            f.close()
        
        return scene,zoneA,zoneC,md
    
    def loadImage(self, filename, factor=1):
        print(filename)
        imgT = cv2.imread(filename)
#         factor = 0.25
        shape = (round(factor * imgT.shape[1]), round(factor * imgT.shape[0]))
        imgMap = np.empty(shape, dtype='uint8')
        imgMap = cv2.resize(imgT, imgMap.shape)
        scene = Scene(imgMap)
        return scene
       
    def calibrate(self,numBoards):
        numBoards = 2#14
        board_w = 10
        board_h = 7
        flag = False
        
        CF = CalibrationFactory(numBoards,board_w,board_h,flag,self.writepath+'/calibration/src/'+str(self.i)+'/',self.writepath+'calibration/' )
        CF.showCamerasPositions()
        
        mtx, dist, rvecs, tvecs = CF.mtx,CF.dist,CF.rvecs,CF.tvecs
        print 'cam'
        print mtx
        print rvecs
        print tvecs
            
        error = CF.reprojectPoints(CF.filenames)
        
#         print 'errors',error
#         print 'image points',CF.imagePoints
        CF.objectPoints
        
        fundamental = cv2.findFundamentalMat(CF.imagePoints[0],CF.imagePoints[1],cv2.FM_8POINT)
        
        retval, H1,H2 = cv2.stereoRectifyUncalibrated(CF.imagePoints[0],CF.imagePoints[1],fundamental[0],CF.shape)
        print 'fundamental', fundamental[0]
        print retval
        print H1
        print H2
        
        fundamental = fundamental[0]
        h, w = CF.shape
#         overlay1 = cv2.warpPerspective(CF.img1, H1, (w, h))
#         f = self.writepath+ 'calibration/1_recified_%d.jpg' % (self.i)
#         print 'savaing to ' + f
#         cv2.imwrite(f, overlay1)
#         overlay = cv2.warpPerspective(CF.img2, H2, (w, h))
#         f = self.writepath+ 'calibration/2_recified_%d.jpg' % (self.i)
#         print 'savaing to ' + f
#         cv2.imwrite(f, overlay)
        
        lines1 = cv2.computeCorrespondEpilines(CF.imagePoints[0],1,fundamental)
        lines1 = lines1.reshape(-1,3)
        
        lines2 = cv2.computeCorrespondEpilines(CF.imagePoints[1],2,fundamental)
        lines2 = lines2.reshape(-1,3)
        
        img2 = CF.img2.copy()
        colors = getColors(len(lines2))
        
        for l,c in zip(lines1,colors):
            self.draw(img2, l, c)
            
        for p,c in zip(CF.imagePoints[1][5:11],colors):
            p = p[0]
            cv2.circle(img2,(p[0],p[1]),15,c,-1)
            
        
            
        f = self.writepath+ 'calibration/%d_epo_2.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, img2)
        
        img1 = CF.img1.copy()
        
        for l,c in zip(lines2,colors):
            self.draw(img1, l, c)
            
        for p,c in zip(CF.imagePoints[0][5:11],colors):
            p = p[0]
            cv2.circle(img1,(p[0],p[1]),15,c,-1)
            
        f = self.writepath+ 'calibration/%d_epo_1.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, img1)
        
        
        
    def draw(self,img,line,color):
        w = img.shape[1]
        r = line
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [w, -(r[2]+r[0]*w)/r[1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color,1)
        
        return img
        
    def detect(self,zone,mask,folder, i, letter):
        pic = i
        
        qubic = QubicObject(zone.image)
        
        #caly bialy
        image = qubic.emptyImage.copy()
        image3 = image.copy()
        
        for kk,wall in qubic.walls.iteritems():
            
            #zaznaczenie powieszhni ściany           
            image3[wall.map == 1] = (255,255,255) 
            for c in wall.contours:
                ll = map(np.array,np.transpose(np.array(c.points)))
                image3[ll] = (255,255,0)
            
#                 mark.drawHoughLines(c.lines, image3, (128,0,128), 1) 
            mark.drawHoughLines(wall.lines, image3, (128,0,128), 1)
                
        for vv in qubic.vertexes:
            cv2.circle(image3,(vv[0],vv[1]),1,(10,0,255),2) 
        if len(qubic.vertexes) > 7:
#             raise Exception('too much vertexes')
            pass
        print 'wierzcholki: ', qubic.vertexes
        fname = self.writepath+'pickle_vertex_%d_%s.p' % (pic,letter)
        f = open(fname,'wb')
        pickle.dump(qubic, f)
        f.close()
        
#         f = open(fname,'rb')
#         obj = pickle.load(f)
        
        f = self.writepath+'%d_lines_%s.jpg' % (pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, image3)
        
        image = zone.image.copy()
        
        for vv in qubic.vertexes:
            cv2.circle(image,(vv[0],vv[1]),1,(10,0,255),2) 
            
        f = self.writepath+'%d_punkty_%s.jpg' % (pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, image)
            
        return qubic.vertexes
    
    def prepareCalibration(self,md):
        img = {}
        image = md.origin
        y = md.middle[1]
        
        img[1] =  Zone(image,0,0,image.shape[1],y)
        img[2] =  Zone(image,0,y,image.shape[1],image.shape[0]-y)
        
        for i in range(1,3):
            f = self.writepath+'/calibration/src/%d/%d.jpg' % (self.i,i)
            print 'savaing to ' + f
            cv2.imwrite(f, img[i].preview)
        
    
    def rrun(self,folder,i):
        self.folder = folder
        self.i = i
        self.writepath = '../img/results/automated/%d/' % folder
        
        scene,zoneA,zoneC,md = self.getZones(folder,i)
        self.prepareCalibration(md)
        
        path1 = self.writepath+ 'pickle_vA_%d.p' % (self.i)
        path2 = self.writepath+ 'pickle_vC_%d.p' % (self.i)
        if os.path.exists(path1) and os.path.exists(path2):
            print 'reading points'
            fname = path1
            f = open(fname,'rb')
            vA = pickle.load(f)
            fname = path2
            f = open(fname,'rb')
            vC = pickle.load(f)
        else:
            ed = edgeDetector.edgeDetector(zoneA.image)
            mask = ed.getSobel()
            vA = self.detect(zoneA,mask,folder, i,'A')
            
            ed = edgeDetector.edgeDetector(zoneC.image)
            mask = ed.getSobel()
            vC = self.detect(zoneC,mask,folder, i,'C')
            
            fname = path1
            f = open(fname,'wb')
            pickle.dump(vA, f)
            f.close()
            
            fname = path2
            f = open(fname,'wb')
            pickle.dump(vC, f)
            f.close()
            
        self.calibrate(2)
    
        
#     def test_10_1(self):
#         self.rrun(10,1)
        
#     def test_10_2(self):
#         self.rrun(10,2)
        
    def test_10_3(self):
        self.rrun(10,3)
        
#     def test_10_4(self):
#         self.rrun(10,4)   
        
       
       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    