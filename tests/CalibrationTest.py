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
from test.test_pep277 import filenames
from numpy.core.numeric import dtype


class edgeDetectionTest(unittest.TestCase):


    def setUp(self):
        np.set_printoptions(precision=4)
        

    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def getFilenames(self,numBoards, basePath):
        '''
        get the names of the sample pictures
        '''
        filenames = {}
        
        for idx in range(numBoards):
            nr = idx + 1
            filenames[idx] =  basePath + str(nr) + ".jpg"
            print basePath + str(nr) + ".jpg"
        
        return filenames
    
    def get3Dpoints(self,board_w,board_n):
        '''
        initialize real 3d points of the chessboard
        '''
        points3D = []
        for i in range(board_n):
            points3D.append([i/board_w,i%board_w,0])
        return points3D
    
    def getMirror3Dpoints(self,board_w,board_n,board_h):
        '''
        initialize real 3d points of the chessboard
        '''
        points3D = []
        for i in range(board_n):
            points3D.append([board_h - 1 - i/board_w, i%board_w , 0])
        return points3D
    
    def findCorners(self,filename,board_size):
        '''
            find corners of each calibration image
        '''
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        found,corners = cv2.findChessboardCorners(gray,board_size,flags=cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS)
    
        if found:
            search_size = (21,21)
#             search_size = (11,11)
            zero_zone = (-1,-1)
            cv2.cornerSubPix(gray,corners,search_size,zero_zone,(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            
        else:
            print 'not found', filename
            
        cv2.drawChessboardCorners(gray, board_size, corners, True);
        cv2.imwrite('calibration/1/1/corners_'+filename[0:-3]+'.jpg',gray)
#         cv2.imshow("corners", gray);
#         cv2.waitKey(0);
        return corners
    
    def calcError(self,imagePoints,imagePointsR):
        '''
            calculates average? error between the real and reprojected points 
        '''
        errorX = 0
        errorY = 0
        numBoards = len(imagePoints)
        board_n = imagePoints[0].shape[0]
        for idx in range(numBoards):    
            for i in range(board_n):
                errorX += abs(imagePoints[idx][i][0][0] - imagePointsR[idx][i][0][0])
                errorY += abs(imagePoints[idx][i][0][1] - imagePointsR[idx][i][0][1])
        errorX /= numBoards * board_n
        errorY /= numBoards * board_n
        
        return (errorX,errorY)
    
    def showDifference(self,filenames,imagePoints2,imagePointsR):
        numBoards = len(imagePoints2)
        board_n = imagePoints2[0].shape[0]
        for idx in range(numBoards):
            img = cv2.imread(filenames[idx])
            for i in range(board_n): 
                cv2.circle(img,(imagePoints2[idx][i][0][0],imagePoints2[idx][i][0][1]),5,(0,255,0),-1)
                cv2.circle(img,(imagePointsR[idx][i][0][0],imagePoints2[idx][i][0][1]),6,(0,0,255),2)
#             cv2.imshow("repr",img)
            cv2.imwrite('calibration/1/1/difference'+str(idx)+'.jpg',img)
#             cv2.waitKey(0)
            
    def showDistortion(self,filenames,numBoards,mtx,dist):
        for idx in range(numBoards):
            #image distortion
            img = cv2.imread(filenames[idx])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img2 = cv2.undistort(gray,mtx,dist)
            
            diff = cv2.absdiff(gray,img2)
            
            cv2.imwrite('calibration/1/1/distortion'+str(idx)+'.jpg',diff)
#             cv2.imshow('n',diff)
#             cv2.waitKey(0)
        
    def testCalibration(self):
        numBoards = 9
        board_w = 10
        board_h = 7
        board_size = (board_w,board_h)
        board_n = board_h*board_w
        
        #3D
        objectPoints = []
        
        #2D 
        imagePoints= []
        
        #
        corners = []
        
        # list of images
        basePath = 'calibration/1/frame0'
        filenames = self.getFilenames(numBoards, basePath)
        points3D = self.get3Dpoints(board_w, board_n)
        
        pointsMirror3D = self.getMirror3Dpoints(board_w, board_n,board_h)
        
        success = 0
        for idx in range(numBoards):
            corners = self.findCorners(filenames[idx],board_size)
#             corners = corners[:,:,0]
#             print 'corners',corners
            imagePoints.append(corners)
                    
            if idx % 2 == 1:
                objectPoints.append(points3D)
            else:
                objectPoints.append(pointsMirror3D)
            success = success +1
            print 'success:', success
                  
        print 'corners success:', success, 'out of',numBoards 
        
        objectPoints2 = np.array(objectPoints,'float32')
        imagePoints2 = np.array(imagePoints,'float32')

        img = cv2.imread(filenames[0])
        shape = (img.shape[1],img.shape[0])
        print 'shape',shape
#         print 'objectpoints',objectPoints2
#         print 'imagepoints',imagePoints2
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints2,imagePoints2,shape)
        print ret
        print 'camera', mtx
        print 'distortion', dist
        for idx in range(numBoards):
            print 'camera', idx
            print 'R vector', rvecs[idx]
            print 'T vector', tvecs[idx]
        
        imagePointsR = {}
        jakobian = {}
        
        for idx in range(numBoards):
#             dist = np.array([[]])
            imagePointsR[idx],jakobian[idx] = cv2.projectPoints(objectPoints2[idx],rvecs[idx],tvecs[idx],mtx,dist)
            
            
            
        self.showDifference(filenames,imagePoints2,imagePointsR)
#         self.showDistortion(filenames, numBoards, mtx, dist)
            
        error = self.calcError(imagePoints, imagePointsR)
        
        origin = np.array([[0],[0],[0]])
        
        scale = 5
        transl = 600
        dispImg = np.zeros((1200,1200,3),dtype=np.uint8)
        
        p1 = origin
        p1 *= scale
        p1 += transl
        
        cv2.circle(dispImg,(p1[0],p1[1]),5,(0,255,0),-1)
        
        for idx in range(numBoards):
            b, invertedCameraMatrix = cv2.invert(mtx)
            print invertedCameraMatrix.shape
            
            
            Rotn,jakobian = cv2.Rodrigues(rvecs[idx])
            
            pt2 = np.array([[2304],[1536-4000],[1]])
            
            
            m1 = np.dot(invertedCameraMatrix,pt2)
            pt2 = np.dot ( Rotn , ((m1)+tvecs[idx]) )
            pt2 *= scale
            pt2 += transl
            
            pt3 = np.array([[2304],[1536+4000],[1]])
            
            m2 = np.dot(invertedCameraMatrix,pt3)
            pt3 = np.dot (Rotn , (m2+tvecs[idx]))
            pt3 *= scale
            pt3 += transl
            
            print 'points',pt2,pt3
            cv2.line(dispImg, (pt2[0],pt2[1]),(pt3[0],pt3[1]),(255,0,0), 4)
        cv2.imwrite('calibration/1/1/cameras.jpg',dispImg)
#         cv2.imshow('cam',dispImg)
#         cv2.waitKey()
        
        print 'errors',error
       
       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    