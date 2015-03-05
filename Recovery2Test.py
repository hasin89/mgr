# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
from calculations import triangulation,DecompPMat
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
from numpy import linalg

from drawings.Draw import getColors
from numpy.linalg.linalg import inv

class edgeDetectionTest(unittest.TestCase):

    writepath = ''
    
    def setUp(self):
        np.set_printoptions(precision=6,suppress=True)

    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def loadImage(self, filename, factor=1):
        print(filename)
        imgT = cv2.imread(filename)
#         factor = 0.25
        shape = (round(factor * imgT.shape[1]), round(factor * imgT.shape[0]))
        imgMap = np.empty(shape, dtype='uint8')
        imgMap = cv2.resize(imgT, imgMap.shape)
        scene = Scene(imgMap)
        return scene
    
    def calcHomography(self,mtx,R):
        
        mtxinv = linalg.inv(mtx)
        h = np.dot(mtx,R)
        H = np.dot(h,mtxinv)
        
        return H
    
    def calcRT(self,rvecs,tvecs):
        R01,j = cv2.Rodrigues(rvecs[0])
        R02,j = cv2.Rodrigues(rvecs[1])
        R = np.dot(R02,R01.T)
        T = tvecs[1] - np.dot(R,tvecs[0])
        
        return R,T 
    
    def calcFundamental(self,mtx,R,T):
        mtxinv = linalg.inv(mtx)
        Tx = np.cross(T.T,np.eye(3)).T
        
        a = np.dot(mtxinv.T,Tx)
        b = np.dot(a,R)
        F = np.dot(b,mtxinv)
        
        return F
               
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
    
    def MirrorPoints(self,points):
        ps = points
        mirror = np.zeros(ps.shape)
        for x in range(ps.shape[0]):
            for y in range(ps.shape[1]):
                mirror[x][ps.shape[1] - y -1] = ps[x][y]
        return mirror
        
        
    def calcTriangulationError(self,idealPoints,calculatedPoints):
        length = max(idealPoints.shape)
        errorX = 0
        errorY = 0
        errorZ = 0
        counter = 0.0
        idealPoints = idealPoints.reshape(140,3)
        calculatedPoints = calculatedPoints.reshape(144,3)
        
        for ideal,real in zip(idealPoints,calculatedPoints):
            errorX += abs(ideal[0]-real[0])
            errorY += abs(ideal[1]-real[1])
            errorZ += abs(ideal[2]-real[2])
            counter += 1.0
            
        return errorX/counter,errorY/counter,errorZ/counter
        
    def draw(self,img,line,color):
        w = img.shape[1]
        r = line
#         print 'line', line
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [w, -(r[2]+r[0]*w)/r[1] ])
        if r[1] == 0 :
            print 'alarm'
        img = cv2.line(img, (y0,x0), (y1,x1), color,1)
        
        return img
    
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
                errorX += abs(imagePoints[idx][i][0] - imagePointsR[idx][i][0][0])
                errorY += abs(imagePoints[idx][i][1] - imagePointsR[idx][i][0][1])
        errorX /= numBoards * board_n
        errorY /= numBoards * board_n
        
        return (errorX,errorY)
    
    
    def undistortRectify(self,mtx,dist,shape,R1):
        R,jacobian = cv2.Rodrigues(R1)
        newCamera,ret = cv2.getOptimalNewCameraMatrix(mtx,dist,(shape[1],shape[0]),1,(shape[1],shape[0]))
        map1,map2 = cv2.initUndistortRectifyMap(mtx,dist,None,newCamera,(shape[1],shape[0]),5)
        
        
        return map1,map2
    
    def makeWow(self,img1, mtx, dist, rvecs, tvecs,origin):
        axis = np.float32([[40,0,0], [0,40,0], [0,0,40],[-500,0,0],[-240,40,40],[-200,40,40],[-240,0,40],[-200,0,40]]).reshape(-1,3)
        imgpoints,jacobian = cv2.projectPoints(axis, rvecs,tvecs, mtx, dist) 
        print 'wow'
        print imgpoints
        print origin
        img1 = self.drawAxes(img1, origin, imgpoints)
        
    
    def drawAxes(self, img, origin, imgpts):
        corner = tuple(origin.ravel())
        
        cv2.line(img, (corner[1],corner[0]), (int (imgpts[0][0][1]) ,int (imgpts[0][0][0])), (255,0,0), 5)
        cv2.line(img, (corner[1],corner[0]), (int (imgpts[1][0][1]) ,int (imgpts[1][0][0])), (0,255,0), 5)
        cv2.line(img, (corner[1],corner[0]), (int (imgpts[2][0][1]) ,int (imgpts[2][0][0])), (0,0,255), 5)
#         cv2.line(img, (corner[1],corner[0]), (int (imgpts[3][0][1]) ,int (imgpts[3][0][0])), (255,255,255), 5)
#         
#         cv2.circle(img,(int (imgpts[4][0][1]) ,int (imgpts[4][0][0])),2,(255,0,255),-1)
#         cv2.circle(img,(int (imgpts[5][0][1]) ,int (imgpts[5][0][0])),2,(255,0,255),-1)
#         cv2.circle(img,(int (imgpts[6][0][1]) ,int (imgpts[6][0][0])),2,(255,0,255),-1)
#         cv2.circle(img,(int (imgpts[7][0][1]) ,int (imgpts[7][0][0])),2,(255,0,255),-1)
        return img
    
    def calibrate(self,filenames):
        factor = 1
        scene1 = self.loadImage(filenames[0], factor)
        scene2 = self.loadImage(filenames[1], factor)
        scenes = [scene1,scene2]
        shape = (3072,4608)
        print shape
        
        offset1 =  (63,1618)
        left = [[[349, 131], [387, 142], [425, 155], [462, 167], [503, 180], [543, 193], [583, 206], [625, 220], [667, 234], [708, 249]], [[367, 182], [406, 194], [444, 207], [482, 220], [521, 233], [561, 246], [602, 260], [643, 274], [685, 288], [727, 303]], [[387, 233], [423, 245], [462, 258], [501, 271], [540, 284], [579, 298], [619, 312], [661, 327], [702, 341], [744, 357]], [[405, 282], [442, 295], [480, 308], [519, 322], [557, 335], [597, 350], [636, 364], [678, 379], [720, 393], [761, 409]], [[423, 331], [459, 345], [498, 358], [536, 371], [574, 386], [613, 400], [653, 415], [695, 429], [736, 445], [777, 460]], [[441, 378], [478, 393], [515, 406], [553, 420], [591, 435], [630, 450], [669, 464], [711, 480], [752, 495], [793, 511]], [[459, 426], [495, 440], [532, 454], [570, 468], [608, 483], [646, 498], [686, 514], [727, 528], [767, 545], [809, 561]], [[164, 695], [198, 711], [233, 727], [269, 744], [306, 761], [344, 778], [382, 797], [421, 815], [461, 834], [502, 855]], [[210, 662], [244, 677], [280, 693], [316, 709], [353, 726], [391, 743], [429, 761], [469, 778], [509, 797], [550, 817]], [[255, 628], [290, 643], [327, 659], [362, 675], [400, 692], [438, 709], [476, 726], [516, 743], [556, 762], [598, 780]], [[300, 596], [336, 611], [372, 627], [409, 642], [446, 658], [484, 675], [522, 692], [563, 709], [603, 727], [644, 745]], [[345, 564], [381, 579], [417, 594], [453, 610], [491, 625], [529, 642], [568, 658], [609, 675], [649, 693], [691, 712]], [[389, 534], [425, 548], [461, 563], [498, 577], [535, 594], [574, 609], [614, 625], [654, 642], [693, 660], [736, 678]], [[433, 503], [468, 517], [505, 531], [541, 546], [580, 561], [618, 577], [658, 593], [697, 610], [738, 626],[779, 644]]]
        left = offset1 + np.array(left)
        left = self.MirrorPoints(left)
        left_real = [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0], [0.0, 7.0, 0.0], [0.0, 8.0, 0.0], [0.0, 9.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.0], [1.0, 5.0, 0.0], [1.0, 6.0, 0.0], [1.0, 7.0, 0.0], [1.0, 8.0, 0.0], [1.0, 9.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [2.0, 5.0, 0.0], [2.0, 6.0, 0.0], [2.0, 7.0, 0.0], [2.0, 8.0, 0.0], [2.0, 9.0, 0.0]], [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [3.0, 6.0, 0.0], [3.0, 7.0, 0.0], [3.0, 8.0, 0.0], [3.0, 9.0, 0.0]], [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 5.0, 0.0], [4.0, 6.0, 0.0], [4.0, 7.0, 0.0], [4.0, 8.0, 0.0], [4.0, 9.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0], [5.0, 4.0, 0.0], [5.0, 5.0, 0.0], [5.0, 6.0, 0.0], [5.0, 7.0, 0.0], [5.0, 8.0, 0.0], [5.0, 9.0, 0.0]], [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, 2.0, 0.0], [6.0, 3.0, 0.0], [6.0, 4.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [6.0, 7.0, 0.0], [6.0, 8.0, 0.0], [6.0, 9.0, 0.0]], [[7.0, 0.0, 7.0], [7.0, 1.0, 7.0], [7.0, 2.0, 7.0], [7.0, 3.0, 7.0], [7.0, 4.0, 7.0], [7.0, 5.0, 7.0], [7.0, 6.0, 7.0], [7.0, 7.0, 7.0], [7.0, 8.0, 7.0], [7.0, 9.0, 7.0]], [[7.0, 0.0, 6.0], [7.0, 1.0, 6.0], [7.0, 2.0, 6.0], [7.0, 3.0, 6.0], [7.0, 4.0, 6.0], [7.0, 5.0, 6.0], [7.0, 6.0, 6.0], [7.0, 7.0, 6.0], [7.0, 8.0, 6.0], [7.0, 9.0, 6.0]], [[7.0, 0.0, 5.0], [7.0, 1.0, 5.0], [7.0, 2.0, 5.0], [7.0, 3.0, 5.0], [7.0, 4.0, 5.0], [7.0, 5.0, 5.0], [7.0, 6.0, 5.0], [7.0, 7.0, 5.0], [7.0, 8.0, 5.0], [7.0, 9.0, 5.0]], [[7.0, 0.0, 4.0], [7.0, 1.0, 4.0], [7.0, 2.0, 4.0], [7.0, 3.0, 4.0], [7.0, 4.0, 4.0], [7.0, 5.0, 4.0], [7.0, 6.0, 4.0], [7.0, 7.0, 4.0], [7.0, 8.0, 4.0], [7.0, 9.0, 4.0]], [[7.0, 0.0, 3.0], [7.0, 1.0, 3.0], [7.0, 2.0, 3.0], [7.0, 3.0, 3.0], [7.0, 4.0, 3.0], [7.0, 5.0, 3.0], [7.0, 6.0, 3.0], [7.0, 7.0, 3.0], [7.0, 8.0, 3.0], [7.0, 9.0, 3.0]], [[7.0, 0.0, 2.0], [7.0, 1.0, 2.0], [7.0, 2.0, 2.0], [7.0, 3.0, 2.0], [7.0, 4.0, 2.0], [7.0, 5.0, 2.0], [7.0, 6.0, 2.0], [7.0, 7.0, 2.0], [7.0, 8.0, 2.0], [7.0, 9.0, 2.0]], [[7.0, 0.0, 1.0], [7.0, 1.0, 1.0], [7.0, 2.0, 1.0], [7.0, 3.0, 1.0], [7.0, 4.0, 1.0], [7.0, 5.0, 1.0], [7.0, 6.0, 1.0], [7.0, 7.0, 1.0], [7.0, 8.0, 1.0], [7.0, 9.0, 1.0]]]
        left_real = np.multiply(left_real,20)
        
        offset2 = (1425,2241)
        right = [[[602, 200], [669, 236], [736, 274], [805, 313], [876, 352], [948, 393], [1021, 436], [1098, 479], [1176, 523], [1255, 570]], [[603, 260], [669, 296], [735, 334], [803, 373], [872, 413], [944, 455], [1016, 498], [1091, 541], [1168, 585], [1245, 632]], [[604, 318], [668, 356], [734, 394], [800, 433], [869, 473], [939, 515], [1011, 557], [1084, 601], [1159, 646], [1236, 692]], [[604, 376], [668, 412], [732, 451], [797, 491], [865, 531], [934, 573], [1005, 615], [1077, 659], [1151, 704], [1226, 751]], [[605, 431], [667, 468], [731, 507], [795, 547], [861, 588], [929, 630], [999, 672], [1071, 715], [1142, 762], [1218, 807]], [[605, 485], [666, 523], [729, 562], [792, 602], [858, 643], [925, 685], [993, 727], [1064, 770], [1135, 816], [1209, 862]], [[605, 538], [665, 577], [727, 615], [790, 655], [854, 696], [921, 738], [988, 780], [1057, 824], [1128, 869], [1201, 915]], [[230, 916], [289, 961], [350, 1007], [413, 1054], [476, 1103], [542, 1154], [609, 1205], [677, 1258], [748, 1313], [820, 1370]], [[287, 866], [346, 909], [407, 955], [469, 1001], [533, 1049], [598, 1098], [665, 1148], [734, 1200], [805, 1253], [877, 1308]], [[342, 817], [402, 860], [463, 904], [525, 949], [589, 996], [654, 1043], [721, 1092], [790, 1143], [860, 1194], [932, 1248]], [[396, 770], [456, 812], [517, 855], [579, 899], [643, 944], [708, 990], [775, 1038], [844, 1087], [914, 1137], [986, 1189]], [[450, 725], [509, 765], [570, 806], [632, 849], [696, 893], [761, 938], [828, 985], [897, 1033], [967, 1081], [1039, 1132]], [[502, 680], [561, 719], [622, 759], [684, 801], [748, 844], [813, 888], [880, 933], [949, 979], [1019, 1027], [1091, 1076]], [[553, 635], [612, 673], [673, 713], [736, 753], [799, 795], [864, 838], [931, 881], [999, 927], [1069, 974], [1141, 1022]]]
        right = offset2 + np.array(right)
        right_real = [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0], [0.0, 7.0, 0.0], [0.0, 8.0, 0.0], [0.0, 9.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.0], [1.0, 5.0, 0.0], [1.0, 6.0, 0.0], [1.0, 7.0, 0.0], [1.0, 8.0, 0.0], [1.0, 9.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [2.0, 5.0, 0.0], [2.0, 6.0, 0.0], [2.0, 7.0, 0.0], [2.0, 8.0, 0.0], [2.0, 9.0, 0.0]], [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [3.0, 6.0, 0.0], [3.0, 7.0, 0.0], [3.0, 8.0, 0.0], [3.0, 9.0, 0.0]], [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 5.0, 0.0], [4.0, 6.0, 0.0], [4.0, 7.0, 0.0], [4.0, 8.0, 0.0], [4.0, 9.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0], [5.0, 4.0, 0.0], [5.0, 5.0, 0.0], [5.0, 6.0, 0.0], [5.0, 7.0, 0.0], [5.0, 8.0, 0.0], [5.0, 9.0, 0.0]], [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, 2.0, 0.0], [6.0, 3.0, 0.0], [6.0, 4.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [6.0, 7.0, 0.0], [6.0, 8.0, 0.0], [6.0, 9.0, 0.0]], [[7.0, 0.0, 7.0], [7.0, 1.0, 7.0], [7.0, 2.0, 7.0], [7.0, 3.0, 7.0], [7.0, 4.0, 7.0], [7.0, 5.0, 7.0], [7.0, 6.0, 7.0], [7.0, 7.0, 7.0], [7.0, 8.0, 7.0], [7.0, 9.0, 7.0]], [[7.0, 0.0, 6.0], [7.0, 1.0, 6.0], [7.0, 2.0, 6.0], [7.0, 3.0, 6.0], [7.0, 4.0, 6.0], [7.0, 5.0, 6.0], [7.0, 6.0, 6.0], [7.0, 7.0, 6.0], [7.0, 8.0, 6.0], [7.0, 9.0, 6.0]], [[7.0, 0.0, 5.0], [7.0, 1.0, 5.0], [7.0, 2.0, 5.0], [7.0, 3.0, 5.0], [7.0, 4.0, 5.0], [7.0, 5.0, 5.0], [7.0, 6.0, 5.0], [7.0, 7.0, 5.0], [7.0, 8.0, 5.0], [7.0, 9.0, 5.0]], [[7.0, 0.0, 4.0], [7.0, 1.0, 4.0], [7.0, 2.0, 4.0], [7.0, 3.0, 4.0], [7.0, 4.0, 4.0], [7.0, 5.0, 4.0], [7.0, 6.0, 4.0], [7.0, 7.0, 4.0], [7.0, 8.0, 4.0], [7.0, 9.0, 4.0]], [[7.0, 0.0, 3.0], [7.0, 1.0, 3.0], [7.0, 2.0, 3.0], [7.0, 3.0, 3.0], [7.0, 4.0, 3.0], [7.0, 5.0, 3.0], [7.0, 6.0, 3.0], [7.0, 7.0, 3.0], [7.0, 8.0, 3.0], [7.0, 9.0, 3.0]], [[7.0, 0.0, 2.0], [7.0, 1.0, 2.0], [7.0, 2.0, 2.0], [7.0, 3.0, 2.0], [7.0, 4.0, 2.0], [7.0, 5.0, 2.0], [7.0, 6.0, 2.0], [7.0, 7.0, 2.0], [7.0, 8.0, 2.0], [7.0, 9.0, 2.0]], [[7.0, 0.0, 1.0], [7.0, 1.0, 1.0], [7.0, 2.0, 1.0], [7.0, 3.0, 1.0], [7.0, 4.0, 1.0], [7.0, 5.0, 1.0], [7.0, 6.0, 1.0], [7.0, 7.0, 1.0], [7.0, 8.0, 1.0], [7.0, 9.0, 1.0]]]
        right_real = np.multiply(right_real,20)
        
        objectPoints = [np.array(left_real).reshape((140,3))[:70] , np.array(right_real).reshape((140,3))[:70]]
        objectPoints2 = np.array(objectPoints,'float32')
        
        imagePoints = [left.reshape((140,2))[:70], right.reshape((140,2))[:70]]
        imagePoints2 = np.array(imagePoints,'float32')
        
        ret, mtx_init, dist_init, rvecs, tvecs = cv2.calibrateCamera(objectPoints2,imagePoints2,shape)
        print mtx_init
        print dist_init
        print rvecs[0]
        print tvecs[0]
        
        objectPoints = [np.array(left_real).reshape((140,3)) , np.array(right_real).reshape((140,3))]
        objectPoints2 = np.array(objectPoints,'float32')
        
        imagePoints = [np.array(left).reshape((140,2))[:140],np.array(right).reshape((140,2))[:140]]
        imagePoints2 = np.array(imagePoints,'float32')
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints2,imagePoints2,shape,mtx_init,dist_init,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        
        img1 = scene1.view.copy()
        img2 = scene2.view.copy()
        
        self.makeWow(img1, mtx, dist, rvecs[0], tvecs[0],imagePoints2[0][0])
        self.makeWow(img2, mtx, dist, rvecs[1], tvecs[1],imagePoints2[1][0])
        
        oPoints1 = np.array([(431,1174),(479,1311),(385,1290),(338,1156)],dtype='float32')
        oPoints2 = np.array([(1877,1635),(1880,1804),(2037,1872),(2038,1701)],dtype='float32')
        imagePoints3 = np.append(imagePoints2[0], oPoints1, 0)
        imagePoints4 = np.append(imagePoints2[1], oPoints2, 0)
        
        fundamental,mask = cv2.findFundamentalMat(imagePoints2[0],imagePoints2[1],cv2.FM_8POINT)
        print 'F1',fundamental
        
        R,T = self.calcRT(rvecs, tvecs)
        F = self.calcFundamental(mtx, R, T)
        H = self.calcHomography(mtx, R)
        
#         cv2.undistortPoints()
        print 'dist shape', dist.shape
        
        print 'F2',F
        
        t1 = tvecs[0]
        t2 = tvecs[1]
        
        R1,jacob = cv2.Rodrigues(rvecs[0])
        R2,jacob = cv2.Rodrigues(rvecs[1])
        
        P1 = self.calcProjectionMatrix(mtx, R1, t1)
        P2 = self.calcProjectionMatrix(mtx, R2, t2)
        
        imagePoints5= imagePoints3.reshape(1,144,2)
        imagePoints6= imagePoints4.reshape(1,144,2)
        
#         imagePoints5,imagePoints6 = cv2.correctMatches(fundamental,imagePoints5,imagePoints6)
#         print imagePoints3.T.shape
        rrr2 = cv2.triangulatePoints(P1,P2,imagePoints3.T , imagePoints4.T)
#         rrr2 = cv2.triangulatePoints(P1,P2,imagePoints5.T , imagePoints6.T)
#         print rrr2.T
        vfunc = np.vectorize(round)
        points = cv2.convertPointsFromHomogeneous(rrr2.T)
        
        print 'triangulation error', self.calcTriangulationError(left_real,points)
        
        points2 = vfunc(points,4)
        print 'recovered:\n', points2.reshape(144,3)
        points2 = vfunc(points)
        mm = np.multiply(points,0.050)
        print 'real:\n', left_real.reshape(140,3)
        
        oPoints3 = np.array([(478,1313),(432,1174)],dtype='float32')
        oPoints4 = np.array([(1883,1805),(1877,1637)],dtype='float32')
        
        oPoints1 = oPoints1.reshape(1,4,2)
        oPoints2 = oPoints2.reshape(1,4,2)
        
#         oPoints1,oPoints2 = cv2.correctMatches(F,oPoints1,oPoints2)
        
        lines1 = cv2.computeCorrespondEpilines(oPoints1,1,fundamental)
        lines1 = lines1.reshape(-1,3)
        
        lines3 = cv2.computeCorrespondEpilines(oPoints2,2,F)
        lines3 = lines3.reshape(-1,3)
        print 'dsrg' 
        
#         numBoards = len(imagePoints2)
#         imagePointsR = {}
#         jakobian = {}
#         for idx in range(numBoards):
# #             dist = np.array([[]])
#             imagePointsR[idx],jakobian[idx] = cv2.projectPoints(objectPoints2[idx],rvecs[idx],tvecs[idx],mtx,dist)
#         
#         board_n = imagePoints2[0].shape[0]
#         print 'xx'
#         print numBoards
#         print board_n
#         for idx in range(numBoards):
#             
#             img = scenes[idx].view.copy()
#             for i in range(board_n): 
#                 cv2.circle(img,(imagePoints2[idx][i][1],imagePoints2[idx][i][0]),5,(0,255,0),-1)
#                 cv2.circle(img,(imagePointsR[idx][i][0][1],imagePointsR[idx][i][0][0]),6,(0,0,255),2)
#         error = self.calcError(imagePoints2, imagePointsR)
#         print error 
        

        
#         newCamera,ret = cv2.getOptimalNewCameraMatrix(mtx,dist,shape,1,shape)
#         img1 = cv2.undistort(scenes[0].view,mtx,dist,None,newCamera)
        colors = getColors(len(lines1))
        for l,c in zip(lines1,colors):
            
            self.draw(img2, l, c)
#             self.draw(img1, l[0], c)
            
        for l,c in zip(lines3,colors):
            
            self.draw(img1, l, c)
#             self.draw(img2, l[0], c)
        
#         for l in zip(lines4):
#             c = (255,0,0)
#             self.draw(img2, l[0], c)
#             self.draw(img1, l[0], c)
#             
#         for l in zip(lines2):
#             c = (255,0,0)
#             self.draw(img2, l[0], c)
#             self.draw(img1, l[0], c)
#         map1,map2 = self.undistortRectify(mtx, dist, shape, rvecs[0])
#         img1 = cv2.remap(img1,map1,map2,cv2.INTER_LINEAR)
#             
#         map1,map2 = self.undistortRectify(mtx, dist, shape, rvecs[1])
#         img2 = cv2.remap(img2,map1,map2,cv2.INTER_LINEAR)
        
#         cv2.imshow("repr",img1)
        cv2.imwrite(self.writepath+'difference_test_'+str(3)+'.jpg',img1)
        cv2.imwrite(self.writepath+'difference_test_'+str(4)+'.jpg',img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        
    
    
    def rrun(self,folder,i):
        left = [[[349, 131], [387, 142], [425, 155], [462, 167], [503, 180], [543, 193], [583, 206], [625, 220], [667, 234], [708, 249]], [[367, 182], [406, 194], [444, 207], [482, 220], [521, 233], [561, 246], [602, 260], [643, 274], [685, 288], [727, 303]], [[387, 233], [423, 245], [462, 258], [501, 271], [540, 284], [579, 298], [619, 312], [661, 327], [702, 341], [744, 357]], [[405, 282], [442, 295], [480, 308], [519, 322], [557, 335], [597, 350], [636, 364], [678, 379], [720, 393], [761, 409]], [[423, 331], [459, 345], [498, 358], [536, 371], [574, 386], [613, 400], [653, 415], [695, 429], [736, 445], [777, 460]], [[441, 378], [478, 393], [515, 406], [553, 420], [591, 435], [630, 450], [669, 464], [711, 480], [752, 495], [793, 511]], [[459, 426], [495, 440], [532, 454], [570, 468], [608, 483], [646, 498], [686, 514], [727, 528], [767, 545], [809, 561]], [[164, 695], [198, 711], [233, 727], [269, 744], [306, 761], [344, 778], [382, 797], [421, 815], [461, 834], [502, 855]], [[210, 662], [244, 677], [280, 693], [316, 709], [353, 726], [391, 743], [429, 761], [469, 778], [509, 797], [550, 817]], [[255, 628], [290, 643], [327, 659], [362, 675], [400, 692], [438, 709], [476, 726], [516, 743], [556, 762], [598, 780]], [[300, 596], [336, 611], [372, 627], [409, 642], [446, 658], [484, 675], [522, 692], [563, 709], [603, 727], [644, 745]], [[345, 564], [381, 579], [417, 594], [453, 610], [491, 625], [529, 642], [568, 658], [609, 675], [649, 693], [691, 712]], [[389, 534], [425, 548], [461, 563], [498, 577], [535, 594], [574, 609], [614, 625], [654, 642], [693, 660], [736, 678]], [[433, 503], [468, 517], [505, 531], [541, 546], [580, 561], [618, 577], [658, 593], [697, 610], [738, 626], [779, 644]]]
        left_real = [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0], [0.0, 7.0, 0.0], [0.0, 8.0, 0.0], [0.0, 9.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.0], [1.0, 5.0, 0.0], [1.0, 6.0, 0.0], [1.0, 7.0, 0.0], [1.0, 8.0, 0.0], [1.0, 9.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [2.0, 5.0, 0.0], [2.0, 6.0, 0.0], [2.0, 7.0, 0.0], [2.0, 8.0, 0.0], [2.0, 9.0, 0.0]], [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [3.0, 6.0, 0.0], [3.0, 7.0, 0.0], [3.0, 8.0, 0.0], [3.0, 9.0, 0.0]], [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 5.0, 0.0], [4.0, 6.0, 0.0], [4.0, 7.0, 0.0], [4.0, 8.0, 0.0], [4.0, 9.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0], [5.0, 4.0, 0.0], [5.0, 5.0, 0.0], [5.0, 6.0, 0.0], [5.0, 7.0, 0.0], [5.0, 8.0, 0.0], [5.0, 9.0, 0.0]], [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, 2.0, 0.0], [6.0, 3.0, 0.0], [6.0, 4.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [6.0, 7.0, 0.0], [6.0, 8.0, 0.0], [6.0, 9.0, 0.0]], [[7.0, 0.0, 7.0], [7.0, 1.0, 7.0], [7.0, 2.0, 7.0], [7.0, 3.0, 7.0], [7.0, 4.0, 7.0], [7.0, 5.0, 7.0], [7.0, 6.0, 7.0], [7.0, 7.0, 7.0], [7.0, 8.0, 7.0], [7.0, 9.0, 7.0]], [[7.0, 0.0, 6.0], [7.0, 1.0, 6.0], [7.0, 2.0, 6.0], [7.0, 3.0, 6.0], [7.0, 4.0, 6.0], [7.0, 5.0, 6.0], [7.0, 6.0, 6.0], [7.0, 7.0, 6.0], [7.0, 8.0, 6.0], [7.0, 9.0, 6.0]], [[7.0, 0.0, 5.0], [7.0, 1.0, 5.0], [7.0, 2.0, 5.0], [7.0, 3.0, 5.0], [7.0, 4.0, 5.0], [7.0, 5.0, 5.0], [7.0, 6.0, 5.0], [7.0, 7.0, 5.0], [7.0, 8.0, 5.0], [7.0, 9.0, 5.0]], [[7.0, 0.0, 4.0], [7.0, 1.0, 4.0], [7.0, 2.0, 4.0], [7.0, 3.0, 4.0], [7.0, 4.0, 4.0], [7.0, 5.0, 4.0], [7.0, 6.0, 4.0], [7.0, 7.0, 4.0], [7.0, 8.0, 4.0], [7.0, 9.0, 4.0]], [[7.0, 0.0, 3.0], [7.0, 1.0, 3.0], [7.0, 2.0, 3.0], [7.0, 3.0, 3.0], [7.0, 4.0, 3.0], [7.0, 5.0, 3.0], [7.0, 6.0, 3.0], [7.0, 7.0, 3.0], [7.0, 8.0, 3.0], [7.0, 9.0, 3.0]], [[7.0, 0.0, 2.0], [7.0, 1.0, 2.0], [7.0, 2.0, 2.0], [7.0, 3.0, 2.0], [7.0, 4.0, 2.0], [7.0, 5.0, 2.0], [7.0, 6.0, 2.0], [7.0, 7.0, 2.0], [7.0, 8.0, 2.0], [7.0, 9.0, 2.0]], [[7.0, 0.0, 1.0], [7.0, 1.0, 1.0], [7.0, 2.0, 1.0], [7.0, 3.0, 1.0], [7.0, 4.0, 1.0], [7.0, 5.0, 1.0], [7.0, 6.0, 1.0], [7.0, 7.0, 1.0], [7.0, 8.0, 1.0], [7.0, 9.0, 1.0]]]
        
        right = [[[602, 200], [669, 236], [736, 274], [805, 313], [876, 352], [948, 393], [1021, 436], [1098, 479], [1176, 523], [1255, 570]], [[603, 260], [669, 296], [735, 334], [803, 373], [872, 413], [944, 455], [1016, 498], [1091, 541], [1168, 585], [1245, 632]], [[604, 318], [668, 356], [734, 394], [800, 433], [869, 473], [939, 515], [1011, 557], [1084, 601], [1159, 646], [1236, 692]], [[604, 376], [668, 412], [732, 451], [797, 491], [865, 531], [934, 573], [1005, 615], [1077, 659], [1151, 704], [1226, 751]], [[605, 431], [667, 468], [731, 507], [795, 547], [861, 588], [929, 630], [999, 672], [1071, 715], [1142, 762], [1218, 807]], [[605, 485], [666, 523], [729, 562], [792, 602], [858, 643], [925, 685], [993, 727], [1064, 770], [1135, 816], [1209, 862]], [[605, 538], [665, 577], [727, 615], [790, 655], [854, 696], [921, 738], [988, 780], [1057, 824], [1128, 869], [1201, 915]], [[230, 916], [289, 961], [350, 1007], [413, 1054], [476, 1103], [542, 1154], [609, 1205], [677, 1258], [748, 1313], [820, 1370]], [[287, 866], [346, 909], [407, 955], [469, 1001], [533, 1049], [598, 1098], [665, 1148], [734, 1200], [805, 1253], [877, 1308]], [[342, 817], [402, 860], [463, 904], [525, 949], [589, 996], [654, 1043], [721, 1092], [790, 1143], [860, 1194], [932, 1248]], [[396, 770], [456, 812], [517, 855], [579, 899], [643, 944], [708, 990], [775, 1038], [844, 1087], [914, 1137], [986, 1189]], [[450, 725], [509, 765], [570, 806], [632, 849], [696, 893], [761, 938], [828, 985], [897, 1033], [967, 1081], [1039, 1132]], [[502, 680], [561, 719], [622, 759], [684, 801], [748, 844], [813, 888], [880, 933], [949, 979], [1019, 1027], [1091, 1076]], [[553, 635], [612, 673], [673, 713], [736, 753], [799, 795], [864, 838], [931, 881], [999, 927], [1069, 974], [1141, 1022]]]
        right_real = [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0], [0.0, 7.0, 0.0], [0.0, 8.0, 0.0], [0.0, 9.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.0], [1.0, 5.0, 0.0], [1.0, 6.0, 0.0], [1.0, 7.0, 0.0], [1.0, 8.0, 0.0], [1.0, 9.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [2.0, 5.0, 0.0], [2.0, 6.0, 0.0], [2.0, 7.0, 0.0], [2.0, 8.0, 0.0], [2.0, 9.0, 0.0]], [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [3.0, 6.0, 0.0], [3.0, 7.0, 0.0], [3.0, 8.0, 0.0], [3.0, 9.0, 0.0]], [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 5.0, 0.0], [4.0, 6.0, 0.0], [4.0, 7.0, 0.0], [4.0, 8.0, 0.0], [4.0, 9.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0], [5.0, 4.0, 0.0], [5.0, 5.0, 0.0], [5.0, 6.0, 0.0], [5.0, 7.0, 0.0], [5.0, 8.0, 0.0], [5.0, 9.0, 0.0]], [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, 2.0, 0.0], [6.0, 3.0, 0.0], [6.0, 4.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [6.0, 7.0, 0.0], [6.0, 8.0, 0.0], [6.0, 9.0, 0.0]], [[7.0, 0.0, 7.0], [7.0, 1.0, 7.0], [7.0, 2.0, 7.0], [7.0, 3.0, 7.0], [7.0, 4.0, 7.0], [7.0, 5.0, 7.0], [7.0, 6.0, 7.0], [7.0, 7.0, 7.0], [7.0, 8.0, 7.0], [7.0, 9.0, 7.0]], [[7.0, 0.0, 6.0], [7.0, 1.0, 6.0], [7.0, 2.0, 6.0], [7.0, 3.0, 6.0], [7.0, 4.0, 6.0], [7.0, 5.0, 6.0], [7.0, 6.0, 6.0], [7.0, 7.0, 6.0], [7.0, 8.0, 6.0], [7.0, 9.0, 6.0]], [[7.0, 0.0, 5.0], [7.0, 1.0, 5.0], [7.0, 2.0, 5.0], [7.0, 3.0, 5.0], [7.0, 4.0, 5.0], [7.0, 5.0, 5.0], [7.0, 6.0, 5.0], [7.0, 7.0, 5.0], [7.0, 8.0, 5.0], [7.0, 9.0, 5.0]], [[7.0, 0.0, 4.0], [7.0, 1.0, 4.0], [7.0, 2.0, 4.0], [7.0, 3.0, 4.0], [7.0, 4.0, 4.0], [7.0, 5.0, 4.0], [7.0, 6.0, 4.0], [7.0, 7.0, 4.0], [7.0, 8.0, 4.0], [7.0, 9.0, 4.0]], [[7.0, 0.0, 3.0], [7.0, 1.0, 3.0], [7.0, 2.0, 3.0], [7.0, 3.0, 3.0], [7.0, 4.0, 3.0], [7.0, 5.0, 3.0], [7.0, 6.0, 3.0], [7.0, 7.0, 3.0], [7.0, 8.0, 3.0], [7.0, 9.0, 3.0]], [[7.0, 0.0, 2.0], [7.0, 1.0, 2.0], [7.0, 2.0, 2.0], [7.0, 3.0, 2.0], [7.0, 4.0, 2.0], [7.0, 5.0, 2.0], [7.0, 6.0, 2.0], [7.0, 7.0, 2.0], [7.0, 8.0, 2.0], [7.0, 9.0, 2.0]], [[7.0, 0.0, 1.0], [7.0, 1.0, 1.0], [7.0, 2.0, 1.0], [7.0, 3.0, 1.0], [7.0, 4.0, 1.0], [7.0, 5.0, 1.0], [7.0, 6.0, 1.0], [7.0, 7.0, 1.0], [7.0, 8.0, 1.0], [7.0, 9.0, 1.0]]]

        self.writepath = '../img/results/automated/%d/' % folder
        
        filename1 = self.writepath+'calibration/src/%d/1.jpg' % (i)
        filename2 = self.writepath+'calibration/src/%d/2.jpg' % (i)
        filenames = [filename1,filename2]
        self.calibrate(filenames)
        
        
    def test_10_7(self):
        self.rrun(10,7)
       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    