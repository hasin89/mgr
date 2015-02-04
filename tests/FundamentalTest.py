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
from numpy.linalg.linalg import inv

class edgeDetectionTest(unittest.TestCase):

    writepath = ''
    
    def setUp(self):
        np.set_printoptions(precision=4)

    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def calibrate(self,numBoards,vA,vC):
        numBoards = 2#14
        board_w = 10
        board_h = 7
        flag = False
        
        CF = CalibrationFactory(numBoards,board_w,board_h,flag,self.writepath+'/calibration/src/'+str(self.i)+'/',self.writepath+'calibration/' )
        CF.showCamerasPositions()
#         CF.reprojectPoints(CF.filenames)
        print 'next'
        k = 1
        img = cv2.imread(CF.filenames[k])
        r = [[0,0,0],[-12,-1,4]]
        o = np.array( r  ,'float32')
        im,jakobian = cv2.projectPoints(o,CF.rvecs[k],CF.tvecs[k],CF.mtx,CF.dist)
        for p in im:
            cv2.circle(img,(p[0][0],p[0][1]),6,(0,0,255),2)
        cv2.imwrite(self.writepath+'calibration/difference11.jpg',img)
        
        mtx, dist, rvecs, tvecs = CF.mtx,CF.dist,CF.rvecs,CF.tvecs
        print 'camera matrix'
        print mtx
        print rvecs
        print 'k'
        print 'shape',CF.shape
        
        
        
        ll = len(CF.imagePoints[0])
        p1 = [np.array([CF.imagePoints[0]],np.float32).reshape(ll,2)]
        p2 = [np.array([CF.imagePoints[1]],np.float32).reshape(ll,2)]
        
        img3 = cv2.undistort(CF.img1,CF.mtx,CF.dist)
        img4 = cv2.undistort(CF.img2,CF.mtx,CF.dist)
        
        p3 = cv2.undistortPoints(np.array(CF.imagePoints[0]),CF.mtx,CF.dist)
        p4 = cv2.undistortPoints(np.array(CF.imagePoints[1]),CF.mtx,CF.dist)
        
        colors = getColors(len(p3))
                    
        f = self.writepath+ 'calibration/%d_undist_1.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, img3)
        
        f = self.writepath+ 'calibration/%d_undist_2.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, img4)

        print ll        
        op = [np.array([CF.objectPoints[0]],np.float32)]
        
            
#         error = CF.reprojectPoints(CF.filenames)
        
#         print 'errors',error
#         print 'image points',CF.imagePoints
        CF.objectPoints
        
        t1 = CF.tvecs[0]
        t2 = CF.tvecs[1]
        R1,jacob = cv2.Rodrigues(CF.rvecs[0])
        R2,jacob = cv2.Rodrigues(CF.rvecs[1])
        print 'R1',R1
        print 'R2',R2
        
        R = np.dot(R2,R1.T)
        T = t2 - np.dot(R.T,t1)
        
        P1 = self.calcProjectionMatrix(mtx, R1, t1)
        P2 = self.calcProjectionMatrix(mtx, R2, t2)
        
        print 'P1',P1
        print 'P2',P2
        
        Te = np.array([[0    , -T[2], T[1]],
                       [T[2] , 0    ,-T[0]],
                       [-T[1], T[0] ,   0]]
                      )
        essential = np.dot(R,Te)
        fundamental1 = np.dot(np.dot(CF.mtx.T,essential),CF.mtx) 
#         print fundamental1
        fundamental,mask = cv2.findFundamentalMat(CF.imagePoints[0],CF.imagePoints[1],cv2.FM_LMEDS)
        
        imagePoints1 = CF.imagePoints[0][mask.ravel()==1]
        imagePoints2 = CF.imagePoints[1][mask.ravel()==1]
        
#         for m,p1,p2 in zip(mask,CF.imagePoints[0],CF.imagePoints[1]):
#             if m == 1:
#                 iPoints1.append(p1)
#                 iPoints2.append(p2)
#                 
#         CF.imagePoints[0] = np.array(iPoints1) 
#         CF.imagePoints[1] = np.array(iPoints2)
        
        retval, H1,H2 = cv2.stereoRectifyUncalibrated(imagePoints1,imagePoints2,fundamental,CF.shape)
        print 'fundamental'
#         print fundamental,'\n'
        
        R1,R2,P11,P22,Q,ROI1,ROI2 = cv2.stereoRectify(CF.mtx,CF.dist,CF.mtx,CF.dist,CF.shape,R,T)
        
        print 'R1',R1
        print 'R2',R2
        
        print 'P1',P11
        print 'P2',P22

        print H1
        print H2
        
        h, w = CF.shape
        
        rrr2 = cv2.triangulatePoints(P1,P2,imagePoints1 , imagePoints2)
#         print rrr2.T
        vfunc = np.vectorize(round)
        points = cv2.convertPointsFromHomogeneous(rrr2.T)
        points2 = vfunc(points)
        print 'recovered:', points2
        
        
        imagePoints2 = np.append(imagePoints2,vC,axis=0).astype(np.float32)
        imagePoints1 = np.append(imagePoints1,vA,axis=0).astype(np.float32)
        
        lines1 = cv2.computeCorrespondEpilines(imagePoints1,1,fundamental1)
        lines1 = lines1.reshape(-1,3)
        
        lines2 = cv2.computeCorrespondEpilines(imagePoints2,2,fundamental1)
        lines2 = lines2.reshape(-1,3)
        
        img2 = CF.img2.copy()
        img1 = CF.img1.copy()
        colors = getColors(len(lines2))
        
        for p1,p2,c in zip(imagePoints1,imagePoints2,colors):
            p1 = p1[0]
            p2 = p2[0]
            cv2.circle(img1,(p1[0],p1[1]),15,c,-1)
            cv2.circle(img2,(p2[0],p2[1]),15,c,-1)
            
        for l,c in zip(lines1,colors):
            self.draw(img2, l, c)
                
        f = self.writepath+ 'calibration/%d_epo_2.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, img2)
        
        
        for l,c in zip(lines2,colors):
            self.draw(img1, l, c)
            
        f = self.writepath+ 'calibration/%d_epo_1.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, img1)
        
        overlay1 = cv2.warpPerspective(img1, H1, (w, h))
        f = self.writepath+ 'calibration/%d_1_recified_.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, overlay1)
        
        overlay = cv2.warpPerspective(img2, H2, (w, h))
        f = self.writepath+ 'calibration/%d_2_recified_.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, overlay)
        
    def calcProjectionMatrix(self,mtx,R1,Tvec):
        a = R1[0,:].tolist()
        a.append(Tvec[0])
        b = R1[1,:].tolist()
        b.append(Tvec[1])
        c = R1[2,:].tolist()
        c.append(Tvec[2])
        
        Me1 = np.matrix([a,b,c])
        print Tvec 
        print Me1       
        P1 = np.dot(mtx,Me1)
            
        return P1    
        
        
    def draw(self,img,line,color):
        w = img.shape[1]
        r = line
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [w, -(r[2]+r[0]*w)/r[1] ])
        if r[1] == 0 :
            print 'alarm'
        img = cv2.line(img, (x0,y0), (x1,y1), color,1)
        
        return img
    
    
    def rrun(self,folder,i):
        self.folder = folder
        self.i = i
        self.writepath = '../img/results/automated/%d/' % folder
        
        numBoards = 2#14
        board_w = 10
        board_h = 7
        flag = False
        
        CF = CalibrationFactory(numBoards,board_w,board_h,flag,self.writepath+'/calibration/src/'+str(self.i)+'/',self.writepath+'calibration/' )
        diffs = CF.showDistortion(CF.filenames, numBoards, CF.mtx, CF.dist)
        
        path1 = self.writepath+ 'pickle_%d_vA.p' % (self.i)
        path2 = self.writepath+ 'pickle_%d_vC.p' % (self.i)
        if os.path.exists(path1) and os.path.exists(path2):
            print 'reading points'
            fname = path1
            f = open(fname,'rb')
            vA = pickle.load(f)
            fname = path2
            f = open(fname,'rb')
            vC = pickle.load(f)
        else:
            raise Exception('pickles not found')
        
        path1 = self.writepath+ 'pickle_%d_zone_A.p' % (self.i)
        path2 = self.writepath+ 'pickle_%d_zone_C.p' % (self.i)
        if os.path.exists(path1) and os.path.exists(path2):
            print 'reading'
            fname = path1
            f = open(fname,'rb')
            zoneA = pickle.load(f)
            f.close()
            fname = path2
            f = open(fname,'rb')
            zoneC = pickle.load(f)
            f.close()
            
        globalVA = []
        for (x,y) in vA:
            p = [(x+zoneA.offsetX,y+zoneA.offsetY)]
            globalVA.append(p)
            
        globalVC = []
        for (x,y) in vC:
            p = [(x+zoneC.offsetX,y+zoneC.offsetY)]
            globalVC.append(p)
            
        self.calibrate(2,np.array(globalVA),np.array(globalVC))
    
        
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
    