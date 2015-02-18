#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import numpy as np
import cv2


'''
Created on Jan 27, 2015

@author: Tomasz
'''

class CalibrationFactory(object):
    '''
    classdocs
    '''
    
    #3D
    objectPoints = None
    
    #2D
    imagePoints = None
    
    filenames = None

    def __init__(self,numBoards=9,w=10,h=7,normalFlag=True,readpath='calibration/1/frame0',writepath='calibration/1/1/'):
        '''
        normalFlag - czy na zdjeciach jest pojedyncza szachownica (True)
        czy sa obie: normalna i odbita -> false
        w drugim przypadku nieparzyste numery plikow powinny zawierac normalna
        w drugim przypadku parzyste numery plikow powinny zawierac odbita
        '''
        self.writepath = writepath
        self.readpath = readpath
        
        self.numBoards = numBoards
        self.board_w = w
        self.board_h = h
        
        self.board_n = self.board_h*self.board_w
        self.board_size = (self.board_w,self.board_h)
        
        filenames = self.getFilenames(numBoards, self.readpath)
        self.img1 = img = cv2.imread(filenames[0])
        self.img2 = cv2.imread(filenames[1])
        
        self.shape = (img.shape[1],img.shape[0])
        
        points3D = self.get3Dpoints(self.board_w, self.board_n)
        print points3D
#         pointsMirror3D = self.getMirror3Dpoints(self.board_w, self.board_n,self.board_h)
        
        self.findPoints(filenames, points3D, normalFlag)
        
        img = self.img1.copy()
        idx = 0
        for p1 in self.imagePoints[idx]:
            print p1
            cv2.circle(img,(p1[0],p1[1]),5,(0,255,0),-1)
        
        cv2.imwrite(self.writepath+'points'+str(idx)+'.jpg',img)
        
        
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = self.calibrate(filenames)
        
        
    def getFilenames(self,numBoards, basePath):
        '''
        get the names of the sample pictures
        '''
        filenames = {}
        
        for idx in range(numBoards):
            nr = idx + 1
            filenames[idx] =  basePath + str(nr) + ".jpg"
        self.filenames = filenames
        return filenames    
        
    def showCamerasPositions(self,scale = 5,trans1 = 600, size = 1200, width = 4000):
        origin = np.array([[0],[0],[0]])
        
        scale = 5
        transl = 600
        dispImg = np.zeros((size,size,3),dtype=np.uint8)
        
        p1 = origin
        p1 *= scale
        p1 += transl
        
        cv2.circle(dispImg,(p1[0],p1[1]),5,(0,255,0),-1)
        
        for idx in range(self.numBoards):
            b, invertedCameraMatrix = cv2.invert(self.mtx)
            
            Rotn,jakobian = cv2.Rodrigues(self.rvecs[idx])
            
            pt2 = np.array([[int(self.shape[0]/2)],[int(self.shape[1]/2)-4000],[1]])
            
            
            m1 = np.dot(invertedCameraMatrix,pt2)
            pt2 = np.dot ( Rotn , ((m1)+self.tvecs[idx]) )
            pt2 *= scale
            pt2 += transl
            
            pt3 = np.array([[int(self.shape[0]/2)],[int(self.shape[1]/2)+4000],[1]])
            
            m2 = np.dot(invertedCameraMatrix,pt3)
            pt3 = np.dot (Rotn , (m2+self.tvecs[idx]))
            pt3 *= scale
            pt3 += transl
            
            cv2.line(dispImg, (pt2[0],pt2[1]),(pt3[0],pt3[1]),(255,0,0), 4)
        cv2.imwrite(self.writepath+ 'cameras.jpg',dispImg)
        
    def calibrate(self,filenames):
        mtx = np.array(
                       [[  4.4e+03,   0.0e+00,   2.2e+03],
                        [  0.0e+00,   4.6e+03,   1.9e+03],
                        [  0.0e+00,   0.0e+00,   1.0e+00]]
                       )
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objectPoints,self.imagePoints,self.shape)
        return ret, mtx, dist, rvecs, tvecs
        
    def findPoints(self,filenames,points3D,normalFlag = True):
        success = 0
        numBoards = self.numBoards
        board_size = self.board_size
        
        #3D
        objectPoints = []
        #2D 
        imagePoints= [] 
        
        for idx in range(numBoards):
            corners = self.findCorners(filenames[idx],board_size)
#             corners = corners[:,:,0]
#             print 'corners',corners
            if idx % 2 == 1 or normalFlag == True:
                imagePoints.append(corners)
            else:
                corners = self.getMirroredCornerPoints(corners)
                imagePoints.append(corners)
            
#             if idx % 2 == 1 or normalFlag == True:
            objectPoints.append(points3D)
#             else:
#                 objectPoints.append(pointsMirror3D)
            success = success +1
            print 'success:', success
                  
        print 'corners success:', success, 'out of',numBoards
        
        objectPoints2 = np.array(objectPoints,'float32')
        imagePoints2 = np.array(imagePoints,'float32')
        
        self.imagePoints = imagePoints2
        self.objectPoints = objectPoints2 

    
    def get3Dpoints(self,board_w,board_n):
        '''
        initialize real 3d points of the chessboard
        '''
        points3D = []
        for i in range(board_n):
            points3D.append([i%board_w,i/board_w,0])
            
        return points3D    
    
    def getMirror3Dpoints(self,board_w,board_n,board_h):
        '''
        initialize real 3d points of the mirrored chessboard
        '''
        points3D = []
        for i in range(board_n):
            points3D.append([board_h - 1 - i/board_w, board_w-i%board_w , 0])
        print 'mirror',points3D
        return points3D
    
    def getMirroredCornerPoints(self,corners):
        cornersTMP = corners.reshape(self.board_h,self.board_w,-1)
        #odwroc kolejnosc punktow w wierszu
        cornersTMP = np.flipud(cornersTMP)
        corners = cornersTMP.reshape(self.board_n,1,2)
        return corners
    
    def reprojectPoints(self,filenames):
        imagePointsR = {}
        jakobian = {}
        for idx in range(self.numBoards):
#             dist = np.array([[]])
            imagePointsR[idx],jakobian[idx] = cv2.projectPoints(self.objectPoints[idx],self.rvecs[idx],self.tvecs[idx],self.mtx,self.dist)
        self.showDifference(filenames,self.imagePoints,imagePointsR)
        error = self.calcError(self.imagePoints, imagePointsR)
        return error
    
    def calcError(self,imagePoints,imagePointsR):
        '''
            calculates average? error between the real and reprojected points 
        '''
        errorX = 0
        errorY = 0
        numBoards = len(imagePoints)
        board_n = imagePoints[0].shape[0]
        board_n = 2
        for idx in range(numBoards):    
            for i in range(board_n):
                errorX += abs(imagePoints[idx][i][0][0] - imagePointsR[idx][i][0][0])
                errorY += abs(imagePoints[idx][i][0][1] - imagePointsR[idx][i][0][1])
        errorX /= numBoards * board_n
        errorY /= numBoards * board_n
        
        return (errorX,errorY)
        
    def findCorners(self,filename,board_size):
        '''
            find corners of each calibration image
        '''
        print filename
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
        cv2.imwrite(self.writepath+'corners_'+filename[0:-3]+'.jpg',gray)
#         cv2.imshow("corners", gray);
#         cv2.waitKey(0);
        corners = self.__orderCorners(corners)
        return corners
    
    def __orderCorners(self,corners):
        '''
            uporzadkuj punkty od lewego gornego rogu
            corners: array(x,y)
        '''
        cornersTMP = corners.reshape(self.board_h,self.board_w,-1)
        
        #sprawdz czy punkty wraz z numerem kolumny rosnie X
        if cornersTMP[0,0][0] < cornersTMP[0,-1][0]:
            pass
        else:
            #odwroc kolejnosc punktow w wierszu
            cornersTMP = np.fliplr(cornersTMP)

        #sprawdz czy punkty wraz z numerem wiersza rosnie Y            
        if cornersTMP[0,0][1] < cornersTMP[-1,0][1]:
            pass
        else:
            #odwroc kolejnosc punktow w wierszu
            cornersTMP = np.flipud(cornersTMP)
        corners = cornersTMP.reshape(self.board_n,1,2)
        return corners
        
    def showDifference(self,filenames,imagePoints2,imagePointsR):
        numBoards = len(imagePoints2)
        board_n = imagePoints2[0].shape[0]
        print 'xx'
        print numBoards
        print board_n
        for idx in range(numBoards):
            img = cv2.imread(filenames[idx])
            for i in range(board_n): 
                print (imagePointsR[idx][i][0][0],imagePointsR[idx][i][0][1])
                cv2.circle(img,(imagePoints2[idx][i][0][0],imagePoints2[idx][i][0][1]),5,(0,255,0),-1)
                cv2.circle(img,(imagePointsR[idx][i][0][0],imagePointsR[idx][i][0][1]),6,(0,0,255),2)
#             cv2.imshow("repr",img)
            cv2.imwrite(self.writepath+'difference'+str(idx)+'.jpg',img)
#             cv2.waitKey(0)

    def showDistortion(self,filenames,numBoards,mtx,dist):
        diffs = []
        for idx in range(numBoards):
            #image distortion
            img = cv2.imread(filenames[idx])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img2 = cv2.undistort(gray,mtx,dist)
            
            diff = cv2.absdiff(gray,img2)
            
            cv2.imwrite(self.writepath + 'distortion'+str(idx)+'.jpg',diff)
            diffs.append(diff)
        return diffs
#             cv2.imshow('n',diff)
#             cv2.waitKey(0)
