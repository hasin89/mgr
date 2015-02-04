'''
Created on Sep 27, 2014

@author: Tomasz
'''
from numpy.linalg import inv,pinv
import unittest
import numpy as np
import cv2
from calculations import triangulation,DecompPMat
import func.analise as an
import pickle
import os
from calculations.calibration import CalibrationFactory
from numpy.core.numeric import dtype

class triangulationTest(unittest.TestCase):


    def setUp(self):
        np.set_printoptions(precision=2)


    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def Recovery(self,ipoints,ipoints2,points3D,shape,Cmtx,Rvec,Tvec,dist):
        
        Pt1 = ipoints[0][0]
        Pt2 = ipoints2[0][0]
        points3d_ = points3D
        newpoints = []
        for p in points3d_:
            newpoints.append((p[0],p[1],p[2]))
        print 'new',newpoints
#         for i in range(len(ipoints)):
#             ipoints[i] = map(int,ipoints[i][0])
#         
#         for i in range(len(ipoints2)):
#             ipoints2[i] = map(int,ipoints2[i][0])
#             
        for i in range(len(points3D)):
            points3D[i] = map(int,points3D[i])
        
        Points = []
        i1 = []
        i2 = []
        for i in range(len(ipoints)):
            element = ( (points3D[i][0],points3D[i][1],points3D[i][2]),(ipoints[i][0][0],ipoints[i][0][1]) )
            Points.append(element)
            i1.append((ipoints[i][0][0],ipoints[i][0][1])) 
        Points2 = []
        for i in range(len(ipoints2)):
            element = ( (points3D[i][0],points3D[i][1],points3D[i][2]),(ipoints2[i][0][0],ipoints2[i][0][1]) )
            Points2.append(element)
            i2.append((ipoints2[i][0][0],ipoints2[i][0][1]))
            
        
        
#         P1 = triangulation.calculate_camera_matrix(Points)
#         K,R,t = DecompPMat.decomposeProjectionMatrix(P1)
#         print 'cam1'
#         print K
#         print 'R',R
#         print 'R',cv2.Rodrigues(R)[0]
#         print 't'
#         print t
#         P2 = triangulation.calculate_camera_matrix(Points2)
#         K,R,t = DecompPMat.decomposeProjectionMatrix(P2)
#         print 'cam1'
#         print K
#         print 'R',R
#         print 'R',cv2.Rodrigues(R)[0]
#         print 't'
#         print t
#         
#         print 'wzor'

        mtx = Cmtx
        
        R1 = Rvec[0]
        R2 = Rvec[1]
        print 'R1v', Rvec[0]
        print 'R1', R1
         
        t1 = Tvec[0]
        print 't1',t1
        t2 = Tvec[1]
        print 't2', t2
        
        P1 = self.calcProjectionMatrix(mtx, R1, t1)
        P2 = self.calcProjectionMatrix(mtx, R2, t2)
        print 'projection P1:',P1
        print 'projection P2:',P2
        
        R1,jacob = cv2.Rodrigues(Rvec[0])
        R2,jacob = cv2.Rodrigues(Rvec[1])
        
        print 'R1', R1
        print 'R2', R2
        
#         ooo = points3d_.reshape(points3d_.shape[0],1,3)
#         ooo = cv2.cv.fromarray(np.array(newpoints))
#         print cv2.cv.fromarray(np.array(i2))
        
        ooo = np.array([newpoints] ,np.float32)
        i1 = np.array([i1],np.float32)
        i2 = np.array([i2],np.float32)
        
        imagePoints1 = i1
        imagePoints2 = i2
        objectPoints = ooo
        print len(imagePoints1[0])== len(imagePoints2[0]) == len(objectPoints[0])
        print len(objectPoints)== len(imagePoints2) == len(imagePoints1)
        print len(imagePoints1[0])
        
        if( len(imagePoints1[0])== len(imagePoints2[0]) == len(objectPoints[0]) and len(objectPoints)== len(imagePoints2) == len(imagePoints1) ) :
            print 'ok'
        else:
#             raise Exception('wrong format of arrays')
            pass
            
        retval, cm1, dist1, cm2, dist2, RR, TT, EE, FF = cv2.stereoCalibrate(ooo,i1,i2,shape)
        
        print mtx
        R = np.dot(R2,R1.T)
        
        print 'R',R
        
        T = t1 - np.dot(R.T,t2)
        print T
        
#         T = np.dot(R2,-1*t1) + t1
#         print T
        
        print RR
        print TT
        
        p1 = imagePoints1[0][0]
        p2 = imagePoints2[0][0]
        print p1
        print p2
        
        c = pinv(P1)
        print c
        a = np.dot( c, np.array([p1[0],p1[1],1])) - T
        print a
        b = np.dot(R,a)
        print b
#         cv2.triangulatePoints(3,2,3,2)
#         p2bis = np.dot(P2,b)
        
#         print p2bis
        
        
        dispImg = np.zeros((shape[0],shape[1],3),dtype=np.uint8)
        
#         cv2.imwrite('../img/results/automated/' + 'compare.jpg',dispImg) 
#         return 1
        
        i = 1
        p1i = ipoints[i][0]
        p2i = ipoints2[i][0]
        print 'origin', points3D[i]
        print 'points', p1i,p2i
        vfunc = np.vectorize(round)
        
        print ipoints.shape
        a = np.array([[[1481,  359]]],np.float32)
        b = np.array([ [[1170, 1512]]],np.float32)

        ipoints = np.append(ipoints,a,axis=0)
        ipoints2 = np.append(ipoints2,b,axis=0)
        print ipoints.shape
        
#         points = cv2.convertPointsFromHomogeneous(rrr1.T)
#         points3 = vfunc(points)
#         print 'recovered:', points3
        
        rrr2 = cv2.triangulatePoints(P1,P2,ipoints , ipoints2)
#         print rrr2.T
        points = cv2.convertPointsFromHomogeneous(rrr2.T)
#         points2 = vfunc(points)
        print 'recovered:', points
        points = [(x/s,y/s,z/s) for (x,y,z,s) in rrr2.T]
        print 'recovered:', points
        
        print mtx
        
        #reprojeckja
#         k = 1
#         img = cv2.imread(self.writepath+'/calibration/src/1.jpg')
#         r = [[0,0,0],[-10,1,0]]
#         o = np.array( r  ,'float32')
#         im,jakobian = cv2.projectPoints(o,CF.rvecs[k],CF.tvecs[k],CF.mtx,CF.dist)
#         for p in im:
#             cv2.circle(img,(p[0][0],p[0][1]),6,(0,0,255),2)
#         cv2.imwrite(self.writepath+'calibration/difference11.jpg',img)
        
        
        
        l = self.calcPosition(Pt1,mtx,R1,t1)
        k = self.calcPosition(Pt2,mtx,R2,t2)
        print 'k',k
        print 'l',l
#         print 'numpy', np.cross([k[0][0],k[1][0],k[2][0] ], [l[0][0],l[1][0],l[2][0]])
        
        middle1 = self.calcPosition( ( int(shape[0]/2),int(shape[1]/2) ) ,mtx,R1,t1)
        o1 = self.calcPosition((0,0),mtx,R1,t1)
        o2 = self.calcPosition((0,0),mtx,R2,t2)
        middle2 = self.calcPosition( (int(shape[0]/2),int(shape[1]/2) ),mtx,R2,t2)
        
        size=  600
        scale = 5
        transpose = 300
        dispImg = np.zeros((size,size,3),dtype=np.uint8)
        i = 1
        for i in range(len(points3D)):
            p1 = np.array([points3D[i][0],points3D[i][1],points3D[i][2]])
            p1 = self.convert(p1, scale, transpose)
            cv2.circle(dispImg,(p1[0],p1[1]),2,(0,255,0),-1)
        
        origin = np.array([0,0,0])
        origin = self.convert(origin, scale, transpose)
        
        
        R1i = inv(R1)
        C1 = np.dot(R1i,t1)
        C1 = C1 * 1
        print 'C1',C1
        R1i = inv(R2)
        C2 = np.dot(R1i,t2)
        print 'C2',C2
        
        l = self.convert(l, scale, transpose)
        k = self.convert(k, scale, transpose)
        t1 = self.convert(C1, scale, transpose)
        print 't1',t1
        t2 = self.convert(C2, scale, transpose)
        print 't2',t2
        middle1 = self.convert(middle1, scale, transpose)
        middle2 = self.convert(middle2, scale, transpose)
        o1 = self.convert(o1, scale, transpose)
        o2 = self.convert(o2, scale, transpose)
        
        cv2.circle(dispImg,(origin[0],origin[1]),5,(0,255,0),-1)
        
        cv2.circle(dispImg,(middle1[0],middle1[1]),3,(0,255,255),-1)
        cv2.circle(dispImg,(middle2[0],middle2[1]),3,(255,255,255),-1)
        
        cv2.circle(dispImg,(C1[0],C1[1]),3,(255,0,255),-1)
        cv2.circle(dispImg,(C2[0],C2[1]),3,(255,0,255),-1)
        
        
        for p in [k,l]:
            print 'drawing',p
#             cv2.circle(dispImg,(int(p[1]),int(p[0])),3,(255,0,255),-1)
        
#         cv2.line(dispImg, (middle1[0],middle1[1]),(t1[0],t1[1]),(255,0,0), 2)
#         cv2.line(dispImg, (middle2[0],middle2[1]),(t2[0],t2[1]),(255,0,0), 2)
            
        cv2.imwrite('../img/results/automated/' + 'cameras.jpg',dispImg)   
        
    def calcProjectionMatrix(self,mtx,Rvec,Tvec):
        R1,jacob = cv2.Rodrigues(Rvec)
        a = R1[0,:].tolist()
        a.append(Tvec[0])
        b = R1[1,:].tolist()
        b.append(Tvec[1])
        c = R1[2,:].tolist()
        c.append(Tvec[2])
        
        Me1 = np.matrix([a,b,c])        
        P1 = np.dot(mtx,Me1)
            
        return P1
        
    def calcPosition(self,ImagePoint,mtx,R,t):
        point = np.array( [ ImagePoint[0], ImagePoint[0], 1 ] ).reshape(3,1)
        b, invertedCameraMatrix = cv2.invert(mtx)  
        # R  -3x3 matrix
        Rotn = R
        m1 = np.dot(invertedCameraMatrix,point)
        pt2 = np.dot ( Rotn , ((m1)+t) )
        return  pt2
    
    def convert(self,point,scale,transpose):
        point *= scale
        point += transpose
        point = map(int,point)
        return point
    
    def test_10_3(self):
        self.rrun(10,3)
        
    def rrun(self,folder,i):
        self.folder = folder
        self.i = i
        self.writepath = '../img/results/automated/%d/' % folder
        
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
            raise Exception('pickles not found')
        
        path1 = self.writepath+ 'pickle_zone_A_%d.p' % (self.i)
        path2 = self.writepath+ 'pickle_zone_C_%d.p' % (self.i)
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
            
        CF = self.calibrate(2,np.array(globalVA),np.array(globalVC))
        CF.showCamerasPositions()
        self.Recovery(CF.imagePoints[0], CF.imagePoints[1], CF.objectPoints[0], CF.shape,CF.mtx, CF.rvecs,CF.tvecs,CF.dist)
        
    def calibrate(self,numBoards,vA,vC):
        numBoards = 2#14
        board_w = 10
        board_h = 7
        flag = False
        
        CF = CalibrationFactory(numBoards,board_w,board_h,flag,self.writepath+'/calibration/src/'+str(self.i)+'/',self.writepath+'calibration/' )
        CF.showCamerasPositions()
        
        return CF
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()