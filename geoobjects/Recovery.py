# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
import operator
import itertools
import numpy as np
import cv2
import func.analise as an
from scene.scene import Scene
from numpy import linalg
import gc

from drawings.Draw import getColors

class thirdDianensionREcovery():

    writepath = ''
    
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
        length1 = max(idealPoints.shape)
        length2 = max(calculatedPoints.shape)
        errorX = 0
        errorY = 0
        errorZ = 0
        counter = 0.0
        idealPoints = idealPoints.reshape(140,3)
        calculatedPoints = calculatedPoints.reshape(length2,3)
        
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
        print 'axis generation'
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
    
    def __sortDictionary(self,x):
        
        sorted_x = sorted(x.items(), key=operator.itemgetter(1))
        
        return sorted_x
    
    def __matchLines(self,p_lines):
        final = {}
        print p_lines
        for k1, line1 in enumerate(p_lines):
            for k2,line2 in enumerate(p_lines):
                if k1==k2:
                    break
                if line1[0][0] == line2[0][0]:
                    v1 = p_lines[k1][0][1]**2 + p_lines[k2][1][1] **2
                    v2 = p_lines[k1][1][1]**2 + p_lines[k2][0][1] **2
                    if v2<v1:
                        final[k1] = p_lines[k1][1][0]
                        final[k2] = p_lines[k1][0][0]
                    if v2>v1:
                        final[k1] = p_lines[k1][0][0]
                        final[k2] = p_lines[k1][1][0]
                        
                else:
                    final[k1] = p_lines[k1][0][0]
        return final
    
    def __matchLinesP(self,oPoints2,lines1):
        SumMin = 100000000
        association = None
        for points in itertools.permutations(oPoints2.tolist()):
            Sum = 0
            for p,line in zip(points,lines1):
                dist = an.calcDistFromLine(p, line)
                dist = dist*dist
                Sum += dist
            if Sum<SumMin:
                SumMin = Sum
                association = points
        return association
    
    def matchPoints(self,oPoints1,oPoints2,lines1,lines2,img1,img2):
        
        colors = getColors(max(len(lines1),len(lines2)))
        for l,c in zip(lines1,colors):
            c = (255,0,0)
            self.draw(img2, l, c)
            
        for l,c in zip(lines2,colors):
            c = (255,0,0)
            self.draw(img1, l, c)

        pairs1 = {}
        association = self.__matchLinesP(oPoints2, lines1)
        for p1,p2 in zip(association, oPoints1):
            pairs1[tuple(p2)] = tuple(p1)
        
        pairs2 = {}
        association = self.__matchLinesP(oPoints1, lines2)
        for p1,p2 in zip(association, oPoints2):
            pairs2[tuple(p1)] = tuple(p2)
            a,b = pairs1[tuple(p1)], tuple(p2)
            if not ( (int(a[0]) == int(b[0])) and (int(a[1]) == int(b[1])) ):
                print 'warning',a,b
#                 raise Exception("points doesn't match")

        colors = getColors( len(pairs1) )
            
        i= 0
        for k,v in pairs1.iteritems():
            c = colors[i]
            k = map(int,k)
            v = map(int,v)
            cv2.circle(img1, (k[1],k[0]), 5, c, -1)
            cv2.circle(img2, (v[1],v[0]), 5, c, -1)
            i += 1     
            
        
        
         
        oPoints1 = np.array(pairs1.keys(),dtype=np.uint32)
        
        oPoints2 =  np.array(pairs1.values(),dtype=np.uint32)
        
#         print oPoints1
#         print oPoints2
        
#          p_lines1 = []
#         for p in oPoints2:    
#             dists = {}
#             for k,line in enumerate(lines1):
#                 dist = an.calcDistFromLine(p, line)
#                 dists[k] = dist
#             dists = self.__sortDictionary(dists)[:2]
#             p_lines1.append(dists)
#         final = self.__matchLines(p_lines1)
#         print final
#         for k,v in final.iteritems():
#             print 'pairs:', oPoints2[k], oPoints1[v]
#             
#         for l,c in zip(lines2,colors):
#             self.draw(img1, l, c)
#         
#         p_lines2 = []
#         for p in oPoints1:    
#             cv2.circle(img1, (p[1],p[0]), 5, (255,0,0), -1) 
#             
#             dists = {}
#             for k,line in enumerate(lines1):
#                 dist = an.calcDistFromLine(p, line)
#                 dists[k] = dist
#             dists = self.__sortDictionary(dists)[:2]
#             p_lines2.append(dists)
#         final = self.__matchLines(p_lines2)
#         print final
#         for k,v in final.iteritems():
#             print 'pairs:',  oPoints2[v], oPoints1[k]
            
        print 'matches in ', 'results/difference_test_'+str(5)+'.jpg'
        cv2.imwrite('results/difference_test_'+str(5)+'.jpg',img1)
        cv2.imwrite('results/difference_test_'+str(6)+'.jpg',img2)
        
        return oPoints1,oPoints2
    
    def calibrateMulti(self,filenames,shape,chessPoints,real,offsets,guess=False,mtx0=None,dist0=None):
        '''
        jedna wspolna lista lista punktow kalibracyjnych wystarcza
        chessPoints - lista punktow dla kazdej z szachownic
        '''
        chess_factor = [20,19.85,20]
        factor = 1
        real = np.multiply(real,chess_factor)
        
        scenes = []
        
        for f in filenames:
            scenes.append( self.loadImage(f, factor) )
            
        points2 = []
        for points,offset in zip(chessPoints,offsets):
            tmp =  offset + np.array(points)
            points2.append( tmp )

        
        objectPoints = []
        imagePoints = []
        
        for points in points2:
            objectPoints.append( np.array(real).reshape((140,3))[:70] )
            imagePoints.append( points.reshape((140,2))[:70] )
        
          
        objectPoints2 = np.array(objectPoints,'float32')
        imagePoints2 = np.array(imagePoints,'float32')
        
        
        if guess:
            objectPoints = []
            imagePoints = []
            for points in points2:
                objectPoints.append( np.array(real).reshape((140,3))[:140] )
                imagePoints.append( points.reshape((140,2))[:140] )
            objectPoints2 = np.array(objectPoints,'float32')
            imagePoints2 = np.array(imagePoints,'float32')
                
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints2,imagePoints2,shape,mtx0,dist0,flags=cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_PRINCIPAL_POINT|cv2.CALIB_FIX_ASPECT_RATIO)
            
        else:
            ret, mtx_init, dist_init, rvecs, tvecs = cv2.calibrateCamera(objectPoints2,imagePoints2,shape)
        
            objectPoints = []
            imagePoints = []
            for points in points2:
                objectPoints.append( np.array(real).reshape((140,3))[:140] )
                imagePoints.append( points.reshape((140,2))[:140] )
            
            objectPoints2 = np.array(objectPoints,'float32')
            imagePoints2 = np.array(imagePoints,'float32')
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints2,imagePoints2,shape,mtx_init,dist_init,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        
        images = []
        for scene in scenes:
            images.append( scene.view)
        
        for k, img in enumerate(images):
            self.makeWow(images[k], mtx, dist, rvecs[k], tvecs[k],imagePoints2[k][0])
        
        return imagePoints2, mtx, dist, rvecs, tvecs, real, images
    
    
#     def calibrateStereo(self,filenames,shape,up,down,up_real,down_real,offsetUp,offsetDown):
#         chess_factor = 20
#         factor = 1
#         scene1 = self.loadImage(filenames[0], factor)
#         scene2 = self.loadImage(filenames[1], factor)
# 
#         offset1 = offsetUp
#         left = up
#         left = offset1 + np.array(left)
# 
#         left_real = up_real
#         left_real = np.multiply(left_real,chess_factor)
#         
#         offset2 = offsetDown
#         right = down
#         right = offset2 + np.array(right)
# 
#         right_real = down_real
#         right_real = np.multiply(right_real,chess_factor)
#         
#         objectPoints = [np.array(left_real).reshape((140,3))[:70] , np.array(right_real).reshape((140,3))[:70]]
#         objectPoints2 = np.array(objectPoints,'float32')
#         
#         imagePoints = [left.reshape((140,2))[:70], right.reshape((140,2))[:70]]
#         imagePoints2 = np.array(imagePoints,'float32')
#         
#         ret, mtx_init, dist_init, rvecs, tvecs = cv2.calibrateCamera(objectPoints2,imagePoints2,shape)
#         
#         objectPoints = [np.array(left_real).reshape((140,3)) , np.array(right_real).reshape((140,3))]
#         objectPoints2 = np.array(objectPoints,'float32')
#         
#         imagePoints = [np.array(left).reshape((140,2))[:140],np.array(right).reshape((140,2))[:140]]
#         imagePoints2 = np.array(imagePoints,'float32')
#         
# #         print objectPoints2
# #         print imagePoints2
#         
#         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints2,imagePoints2,shape,mtx_init,dist_init,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
#         
#         img1 = scene1.view.copy()
#         img2 = scene2.view.copy()
#         
#         self.makeWow(img1, mtx, dist, rvecs[0], tvecs[0],imagePoints2[0][0])
#         self.makeWow(img2, mtx, dist, rvecs[1], tvecs[1],imagePoints2[1][0])
#         
#         images = [img1,img2]
#         
#         return imagePoints2, mtx, dist, rvecs, tvecs, left_real, images
        
        
    def getFundamental(self, imagePoints2, rvecs,tvecs,mtx,dist, real, images, modelPoints=None):
        
        img1 = images[0]
        img2 = images[1]
        
        if modelPoints is not None:
            oPoints1 = np.array(modelPoints[0], dtype='float32')
            oPoints2 = np.array(modelPoints[1], dtype='float32')
            
    #         oPoints1 = np.array([(431,1174),(479,1311),(385,1290),(338,1156)],dtype='float32')
    #         oPoints2 = np.array([(1877,1635),(1880,1804),(2037,1872),(2038,1701)],dtype='float32')
            
            imagePoints3 = np.append(imagePoints2[0], oPoints1, 0)
            imagePoints4 = np.append(imagePoints2[1], oPoints2, 0)
        else:
            imagePoints3 = imagePoints2[0]
            imagePoints4 = imagePoints2[1]
        
        print 'shape img', imagePoints3.T
        
        
#         metoda Hartleya
#         fundamental,mask = cv2.findFundamentalMat(imagePoints2[0],imagePoints2[1],cv2.FM_8POINT)
#         print 'F1',fundamental
        
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
        
#         imagePoints5,imagePoints6 = cv2.correctMatches(fundamental,imagePoints5,imagePoints6)
#         print imagePoints3.T.shape
        
        rrr2 = cv2.triangulatePoints(P1,P2,imagePoints3.T , imagePoints4.T)
#         rrr2 = cv2.triangulatePoints(P1,P2,imagePoints5.T , imagePoints6.T)
#         print rrr2.T
        vfunc = np.vectorize(round)
        points = cv2.convertPointsFromHomogeneous(rrr2.T)
        
        print 'triangulation error', self.calcTriangulationError(real,points)
        length2 = max(points.shape)
        
        points2 = vfunc(points,4)
#         print 'recovered:\n', points2.reshape(length2,3)[-4:]
        points2 = vfunc(points)
#         print 'real:\n', left_real.reshape(140,3)
        
#         self.calculateEpilines(oPoints1, oPoints2, F, img1, img2)
        
        
#         oPoints1 = oPoints1.reshape(1,4,2)
#         oPoints2 = oPoints2.reshape(1,4,2)
#         
# #         oPoints1,oPoints2 = cv2.correctMatches(F,oPoints1,oPoints2)
#         
#         lines1 = cv2.computeCorrespondEpilines(oPoints1,1,F)
#         lines1 = lines1.reshape(-1,3)
#         
#         lines3 = cv2.computeCorrespondEpilines(oPoints2,2,F)
#         lines3 = lines3.reshape(-1,3)
#         
# #         newCamera,ret = cv2.getOptimalNewCameraMatrix(mtx,dist,shape,1,shape)
# #         img1 = cv2.undistort(scenes[0].view,mtx,dist,None,newCamera)
#         colors = getColors(len(lines1))
#         for l,c in zip(lines1,colors):
#             
#             self.draw(img2, l, c)
# #             self.draw(img1, l[0], c)
#             
#         for l,c in zip(lines3,colors):
#             
#             self.draw(img1, l, c)
#         
# #         cv2.imshow("repr",img1)
#         print 'matches in ', 'results/difference_test_'+str(5)+'.jpg'
#         cv2.imwrite('results/difference_test_'+str(5)+'.jpg',img1)
#         cv2.imwrite('results/difference_test_'+str(6)+'.jpg',img2)
        
        return P1,P2, F
    
    def calculateEpilines(self,oPoints1,oPoints2,F,img1,img2):
        n = oPoints1.shape[0]
        oPoints1 = oPoints1.reshape(1,n,2)
        
        m = oPoints2.shape[0]
        oPoints2 = oPoints2.reshape(1,m,2)
        
        print 'puntow', n,m
        
#         oPoints1,oPoints2 = cv2.correctMatches(F,oPoints1,oPoints2)
        
        lines1 = cv2.computeCorrespondEpilines(oPoints1,1,F)
        lines1 = lines1.reshape(-1,3)
        
        lines3 = cv2.computeCorrespondEpilines(oPoints2,2,F)
        lines3 = lines3.reshape(-1,3)
        
#         newCamera,ret = cv2.getOptimalNewCameraMatrix(mtx,dist,shape,1,shape)
#         img1 = cv2.undistort(scenes[0].view,mtx,dist,None,newCamera)
#         colors = getColors(max(len(lines1),len(lines3)))
#         for l,c in zip(lines1,colors):
#             self.draw(img2, l, c)
# 
#         for p in oPoints2[0]:
#             cv2.circle(img2, (p[1],p[0]), 5, (255,0,0), -1) 
# #             self.draw(img1, l[0], c)
#             
#         for l,c in zip(lines3,colors):
#             self.draw(img1, l, c)
#             
#         for p in oPoints1[0]:    
#             cv2.circle(img1, (p[1],p[0]), 5, (255,0,0), -1) 
        
#         cv2.imshow("repr",img1)
#         print 'matches in ', 'results/difference_test_'+str(5)+'.jpg'
#         cv2.imwrite('results/difference_test_'+str(5)+'.jpg',img1)
#         cv2.imwrite('results/difference_test_'+str(6)+'.jpg',img2)
        
        return lines1,lines3
            
        
    
    
    def rrun(self,folder,i):
        left = [[[349, 131], [387, 142], [425, 155], [462, 167], [503, 180], [543, 193], [583, 206], [625, 220], [667, 234], [708, 249]], [[367, 182], [406, 194], [444, 207], [482, 220], [521, 233], [561, 246], [602, 260], [643, 274], [685, 288], [727, 303]], [[387, 233], [423, 245], [462, 258], [501, 271], [540, 284], [579, 298], [619, 312], [661, 327], [702, 341], [744, 357]], [[405, 282], [442, 295], [480, 308], [519, 322], [557, 335], [597, 350], [636, 364], [678, 379], [720, 393], [761, 409]], [[423, 331], [459, 345], [498, 358], [536, 371], [574, 386], [613, 400], [653, 415], [695, 429], [736, 445], [777, 460]], [[441, 378], [478, 393], [515, 406], [553, 420], [591, 435], [630, 450], [669, 464], [711, 480], [752, 495], [793, 511]], [[459, 426], [495, 440], [532, 454], [570, 468], [608, 483], [646, 498], [686, 514], [727, 528], [767, 545], [809, 561]], [[164, 695], [198, 711], [233, 727], [269, 744], [306, 761], [344, 778], [382, 797], [421, 815], [461, 834], [502, 855]], [[210, 662], [244, 677], [280, 693], [316, 709], [353, 726], [391, 743], [429, 761], [469, 778], [509, 797], [550, 817]], [[255, 628], [290, 643], [327, 659], [362, 675], [400, 692], [438, 709], [476, 726], [516, 743], [556, 762], [598, 780]], [[300, 596], [336, 611], [372, 627], [409, 642], [446, 658], [484, 675], [522, 692], [563, 709], [603, 727], [644, 745]], [[345, 564], [381, 579], [417, 594], [453, 610], [491, 625], [529, 642], [568, 658], [609, 675], [649, 693], [691, 712]], [[389, 534], [425, 548], [461, 563], [498, 577], [535, 594], [574, 609], [614, 625], [654, 642], [693, 660], [736, 678]], [[433, 503], [468, 517], [505, 531], [541, 546], [580, 561], [618, 577], [658, 593], [697, 610], [738, 626], [779, 644]]]
        left_real = [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0], [0.0, 7.0, 0.0], [0.0, 8.0, 0.0], [0.0, 9.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.0], [1.0, 5.0, 0.0], [1.0, 6.0, 0.0], [1.0, 7.0, 0.0], [1.0, 8.0, 0.0], [1.0, 9.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [2.0, 5.0, 0.0], [2.0, 6.0, 0.0], [2.0, 7.0, 0.0], [2.0, 8.0, 0.0], [2.0, 9.0, 0.0]], [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [3.0, 6.0, 0.0], [3.0, 7.0, 0.0], [3.0, 8.0, 0.0], [3.0, 9.0, 0.0]], [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 5.0, 0.0], [4.0, 6.0, 0.0], [4.0, 7.0, 0.0], [4.0, 8.0, 0.0], [4.0, 9.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0], [5.0, 4.0, 0.0], [5.0, 5.0, 0.0], [5.0, 6.0, 0.0], [5.0, 7.0, 0.0], [5.0, 8.0, 0.0], [5.0, 9.0, 0.0]], [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, 2.0, 0.0], [6.0, 3.0, 0.0], [6.0, 4.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [6.0, 7.0, 0.0], [6.0, 8.0, 0.0], [6.0, 9.0, 0.0]], [[7.0, 0.0, 7.0], [7.0, 1.0, 7.0], [7.0, 2.0, 7.0], [7.0, 3.0, 7.0], [7.0, 4.0, 7.0], [7.0, 5.0, 7.0], [7.0, 6.0, 7.0], [7.0, 7.0, 7.0], [7.0, 8.0, 7.0], [7.0, 9.0, 7.0]], [[7.0, 0.0, 6.0], [7.0, 1.0, 6.0], [7.0, 2.0, 6.0], [7.0, 3.0, 6.0], [7.0, 4.0, 6.0], [7.0, 5.0, 6.0], [7.0, 6.0, 6.0], [7.0, 7.0, 6.0], [7.0, 8.0, 6.0], [7.0, 9.0, 6.0]], [[7.0, 0.0, 5.0], [7.0, 1.0, 5.0], [7.0, 2.0, 5.0], [7.0, 3.0, 5.0], [7.0, 4.0, 5.0], [7.0, 5.0, 5.0], [7.0, 6.0, 5.0], [7.0, 7.0, 5.0], [7.0, 8.0, 5.0], [7.0, 9.0, 5.0]], [[7.0, 0.0, 4.0], [7.0, 1.0, 4.0], [7.0, 2.0, 4.0], [7.0, 3.0, 4.0], [7.0, 4.0, 4.0], [7.0, 5.0, 4.0], [7.0, 6.0, 4.0], [7.0, 7.0, 4.0], [7.0, 8.0, 4.0], [7.0, 9.0, 4.0]], [[7.0, 0.0, 3.0], [7.0, 1.0, 3.0], [7.0, 2.0, 3.0], [7.0, 3.0, 3.0], [7.0, 4.0, 3.0], [7.0, 5.0, 3.0], [7.0, 6.0, 3.0], [7.0, 7.0, 3.0], [7.0, 8.0, 3.0], [7.0, 9.0, 3.0]], [[7.0, 0.0, 2.0], [7.0, 1.0, 2.0], [7.0, 2.0, 2.0], [7.0, 3.0, 2.0], [7.0, 4.0, 2.0], [7.0, 5.0, 2.0], [7.0, 6.0, 2.0], [7.0, 7.0, 2.0], [7.0, 8.0, 2.0], [7.0, 9.0, 2.0]], [[7.0, 0.0, 1.0], [7.0, 1.0, 1.0], [7.0, 2.0, 1.0], [7.0, 3.0, 1.0], [7.0, 4.0, 1.0], [7.0, 5.0, 1.0], [7.0, 6.0, 1.0], [7.0, 7.0, 1.0], [7.0, 8.0, 1.0], [7.0, 9.0, 1.0]]]
          
        right = [[[602, 200], [669, 236], [736, 274], [805, 313], [876, 352], [948, 393], [1021, 436], [1098, 479], [1176, 523], [1255, 570]], [[603, 260], [669, 296], [735, 334], [803, 373], [872, 413], [944, 455], [1016, 498], [1091, 541], [1168, 585], [1245, 632]], [[604, 318], [668, 356], [734, 394], [800, 433], [869, 473], [939, 515], [1011, 557], [1084, 601], [1159, 646], [1236, 692]], [[604, 376], [668, 412], [732, 451], [797, 491], [865, 531], [934, 573], [1005, 615], [1077, 659], [1151, 704], [1226, 751]], [[605, 431], [667, 468], [731, 507], [795, 547], [861, 588], [929, 630], [999, 672], [1071, 715], [1142, 762], [1218, 807]], [[605, 485], [666, 523], [729, 562], [792, 602], [858, 643], [925, 685], [993, 727], [1064, 770], [1135, 816], [1209, 862]], [[605, 538], [665, 577], [727, 615], [790, 655], [854, 696], [921, 738], [988, 780], [1057, 824], [1128, 869], [1201, 915]], [[230, 916], [289, 961], [350, 1007], [413, 1054], [476, 1103], [542, 1154], [609, 1205], [677, 1258], [748, 1313], [820, 1370]], [[287, 866], [346, 909], [407, 955], [469, 1001], [533, 1049], [598, 1098], [665, 1148], [734, 1200], [805, 1253], [877, 1308]], [[342, 817], [402, 860], [463, 904], [525, 949], [589, 996], [654, 1043], [721, 1092], [790, 1143], [860, 1194], [932, 1248]], [[396, 770], [456, 812], [517, 855], [579, 899], [643, 944], [708, 990], [775, 1038], [844, 1087], [914, 1137], [986, 1189]], [[450, 725], [509, 765], [570, 806], [632, 849], [696, 893], [761, 938], [828, 985], [897, 1033], [967, 1081], [1039, 1132]], [[502, 680], [561, 719], [622, 759], [684, 801], [748, 844], [813, 888], [880, 933], [949, 979], [1019, 1027], [1091, 1076]], [[553, 635], [612, 673], [673, 713], [736, 753], [799, 795], [864, 838], [931, 881], [999, 927], [1069, 974], [1141, 1022]]]
        right_real = [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0], [0.0, 7.0, 0.0], [0.0, 8.0, 0.0], [0.0, 9.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.0], [1.0, 5.0, 0.0], [1.0, 6.0, 0.0], [1.0, 7.0, 0.0], [1.0, 8.0, 0.0], [1.0, 9.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [2.0, 5.0, 0.0], [2.0, 6.0, 0.0], [2.0, 7.0, 0.0], [2.0, 8.0, 0.0], [2.0, 9.0, 0.0]], [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [3.0, 6.0, 0.0], [3.0, 7.0, 0.0], [3.0, 8.0, 0.0], [3.0, 9.0, 0.0]], [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 5.0, 0.0], [4.0, 6.0, 0.0], [4.0, 7.0, 0.0], [4.0, 8.0, 0.0], [4.0, 9.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0], [5.0, 4.0, 0.0], [5.0, 5.0, 0.0], [5.0, 6.0, 0.0], [5.0, 7.0, 0.0], [5.0, 8.0, 0.0], [5.0, 9.0, 0.0]], [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, 2.0, 0.0], [6.0, 3.0, 0.0], [6.0, 4.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [6.0, 7.0, 0.0], [6.0, 8.0, 0.0], [6.0, 9.0, 0.0]], [[7.0, 0.0, 7.0], [7.0, 1.0, 7.0], [7.0, 2.0, 7.0], [7.0, 3.0, 7.0], [7.0, 4.0, 7.0], [7.0, 5.0, 7.0], [7.0, 6.0, 7.0], [7.0, 7.0, 7.0], [7.0, 8.0, 7.0], [7.0, 9.0, 7.0]], [[7.0, 0.0, 6.0], [7.0, 1.0, 6.0], [7.0, 2.0, 6.0], [7.0, 3.0, 6.0], [7.0, 4.0, 6.0], [7.0, 5.0, 6.0], [7.0, 6.0, 6.0], [7.0, 7.0, 6.0], [7.0, 8.0, 6.0], [7.0, 9.0, 6.0]], [[7.0, 0.0, 5.0], [7.0, 1.0, 5.0], [7.0, 2.0, 5.0], [7.0, 3.0, 5.0], [7.0, 4.0, 5.0], [7.0, 5.0, 5.0], [7.0, 6.0, 5.0], [7.0, 7.0, 5.0], [7.0, 8.0, 5.0], [7.0, 9.0, 5.0]], [[7.0, 0.0, 4.0], [7.0, 1.0, 4.0], [7.0, 2.0, 4.0], [7.0, 3.0, 4.0], [7.0, 4.0, 4.0], [7.0, 5.0, 4.0], [7.0, 6.0, 4.0], [7.0, 7.0, 4.0], [7.0, 8.0, 4.0], [7.0, 9.0, 4.0]], [[7.0, 0.0, 3.0], [7.0, 1.0, 3.0], [7.0, 2.0, 3.0], [7.0, 3.0, 3.0], [7.0, 4.0, 3.0], [7.0, 5.0, 3.0], [7.0, 6.0, 3.0], [7.0, 7.0, 3.0], [7.0, 8.0, 3.0], [7.0, 9.0, 3.0]], [[7.0, 0.0, 2.0], [7.0, 1.0, 2.0], [7.0, 2.0, 2.0], [7.0, 3.0, 2.0], [7.0, 4.0, 2.0], [7.0, 5.0, 2.0], [7.0, 6.0, 2.0], [7.0, 7.0, 2.0], [7.0, 8.0, 2.0], [7.0, 9.0, 2.0]], [[7.0, 0.0, 1.0], [7.0, 1.0, 1.0], [7.0, 2.0, 1.0], [7.0, 3.0, 1.0], [7.0, 4.0, 1.0], [7.0, 5.0, 1.0], [7.0, 6.0, 1.0], [7.0, 7.0, 1.0], [7.0, 8.0, 1.0], [7.0, 9.0, 1.0]]]
  
       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    o = thirdDianensionREcovery()
    filenames = ['../results/calibration/0_1.jpg','../results/calibration/0_2.jpg']
    up_real =   [[120.0, 0.0, 0.0], [120.0, 20.0, 0.0], [120.0, 40.0, 0.0], [120.0, 60.0, 0.0], [120.0, 80.0, 0.0], [120.0, 100.0, 0.0], [120.0, 120.0, 0.0], [120.0, 140.0, 0.0], [120.0, 160.0, 0.0], [120.0, 180.0, 0.0], [100.0, 0.0, 0.0], [100.0, 20.0, 0.0], [100.0, 40.0, 0.0], [100.0, 60.0, 0.0], [100.0, 80.0, 0.0], [100.0, 100.0, 0.0], [100.0, 120.0, 0.0], [100.0, 140.0, 0.0], [100.0, 160.0, 0.0], [100.0, 180.0, 0.0], [80.0, 0.0, 0.0], [80.0, 20.0, 0.0], [80.0, 40.0, 0.0], [80.0, 60.0, 0.0], [80.0, 80.0, 0.0], [80.0, 100.0, 0.0], [80.0, 120.0, 0.0], [80.0, 140.0, 0.0], [80.0, 160.0, 0.0], [80.0, 180.0, 0.0], [60.0, 0.0, 0.0], [60.0, 20.0, 0.0], [60.0, 40.0, 0.0], [60.0, 60.0, 0.0], [60.0, 80.0, 0.0], [60.0, 100.0, 0.0], [60.0, 120.0, 0.0], [60.0, 140.0, 0.0], [60.0, 160.0, 0.0], [60.0, 180.0, 0.0], [40.0, 0.0, 0.0], [40.0, 20.0, 0.0], [40.0, 40.0, 0.0], [40.0, 60.0, 0.0], [40.0, 80.0, 0.0], [40.0, 100.0, 0.0], [40.0, 120.0, 0.0], [40.0, 140.0, 0.0], [40.0, 160.0, 0.0], [40.0, 180.0, 0.0], [20.0, 0.0, 0.0], [20.0, 20.0, 0.0], [20.0, 40.0, 0.0], [20.0, 60.0, 0.0], [20.0, 80.0, 0.0], [20.0, 100.0, 0.0], [20.0, 120.0, 0.0], [20.0, 140.0, 0.0], [20.0, 160.0, 0.0], [20.0, 180.0, 0.0], [0.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 40.0, 0.0], [0.0, 60.0, 0.0], [0.0, 80.0, 0.0], [0.0, 100.0, 0.0], [0.0, 120.0, 0.0], [0.0, 140.0, 0.0], [0.0, 160.0, 0.0], [0.0, 180.0, 0.0], [140.0, 0.0, 20.0], [140.0, 20.0, 20.0], [140.0, 40.0, 20.0], [140.0, 60.0, 20.0], [140.0, 80.0, 20.0], [140.0, 100.0, 20.0], [140.0, 120.0, 20.0], [140.0, 140.0, 20.0], [140.0, 160.0, 20.0], [140.0, 180.0, 20.0], [140.0, 0.0, 40.0], [140.0, 20.0, 40.0], [140.0, 40.0, 40.0], [140.0, 60.0, 40.0], [140.0, 80.0, 40.0], [140.0, 100.0, 40.0], [140.0, 120.0, 40.0], [140.0, 140.0, 40.0], [140.0, 160.0, 40.0], [140.0, 180.0, 40.0], [140.0, 0.0, 60.0], [140.0, 20.0, 60.0], [140.0, 40.0, 60.0], [140.0, 60.0, 60.0], [140.0, 80.0, 60.0], [140.0, 100.0, 60.0], [140.0, 120.0, 60.0], [140.0, 140.0, 60.0], [140.0, 160.0, 60.0], [140.0, 180.0, 60.0], [140.0, 0.0, 80.0], [140.0, 20.0, 80.0], [140.0, 40.0, 80.0], [140.0, 60.0, 80.0], [140.0, 80.0, 80.0], [140.0, 100.0, 80.0], [140.0, 120.0, 80.0], [140.0, 140.0, 80.0], [140.0, 160.0, 80.0], [140.0, 180.0, 80.0], [140.0, 0.0, 100.0], [140.0, 20.0, 100.0], [140.0, 40.0, 100.0], [140.0, 60.0, 100.0], [140.0, 80.0, 100.0], [140.0, 100.0, 100.0], [140.0, 120.0, 100.0], [140.0, 140.0, 100.0], [140.0, 160.0, 100.0], [140.0, 180.0, 100.0], [140.0, 0.0, 120.0], [140.0, 20.0, 120.0], [140.0, 40.0, 120.0], [140.0, 60.0, 120.0], [140.0, 80.0, 120.0], [140.0, 100.0, 120.0], [140.0, 120.0, 120.0], [140.0, 140.0, 120.0], [140.0, 160.0, 120.0], [140.0, 180.0, 120.0], [140.0, 0.0, 140.0], [140.0, 20.0, 140.0], [140.0, 40.0, 140.0], [140.0, 60.0, 140.0], [140.0, 80.0, 140.0], [140.0, 100.0, 140.0], [140.0, 120.0, 140.0], [140.0, 140.0, 140.0], [140.0, 160.0, 140.0], [140.0, 180.0, 140.0]]
    down_real = [[120.0, 0.0, 0.0], [120.0, 20.0, 0.0], [120.0, 40.0, 0.0], [120.0, 60.0, 0.0], [120.0, 80.0, 0.0], [120.0, 100.0, 0.0], [120.0, 120.0, 0.0], [120.0, 140.0, 0.0], [120.0, 160.0, 0.0], [120.0, 180.0, 0.0], [100.0, 0.0, 0.0], [100.0, 20.0, 0.0], [100.0, 40.0, 0.0], [100.0, 60.0, 0.0], [100.0, 80.0, 0.0], [100.0, 100.0, 0.0], [100.0, 120.0, 0.0], [100.0, 140.0, 0.0], [100.0, 160.0, 0.0], [100.0, 180.0, 0.0], [80.0, 0.0, 0.0], [80.0, 20.0, 0.0], [80.0, 40.0, 0.0], [80.0, 60.0, 0.0], [80.0, 80.0, 0.0], [80.0, 100.0, 0.0], [80.0, 120.0, 0.0], [80.0, 140.0, 0.0], [80.0, 160.0, 0.0], [80.0, 180.0, 0.0], [60.0, 0.0, 0.0], [60.0, 20.0, 0.0], [60.0, 40.0, 0.0], [60.0, 60.0, 0.0], [60.0, 80.0, 0.0], [60.0, 100.0, 0.0], [60.0, 120.0, 0.0], [60.0, 140.0, 0.0], [60.0, 160.0, 0.0], [60.0, 180.0, 0.0], [40.0, 0.0, 0.0], [40.0, 20.0, 0.0], [40.0, 40.0, 0.0], [40.0, 60.0, 0.0], [40.0, 80.0, 0.0], [40.0, 100.0, 0.0], [40.0, 120.0, 0.0], [40.0, 140.0, 0.0], [40.0, 160.0, 0.0], [40.0, 180.0, 0.0], [20.0, 0.0, 0.0], [20.0, 20.0, 0.0], [20.0, 40.0, 0.0], [20.0, 60.0, 0.0], [20.0, 80.0, 0.0], [20.0, 100.0, 0.0], [20.0, 120.0, 0.0], [20.0, 140.0, 0.0], [20.0, 160.0, 0.0], [20.0, 180.0, 0.0], [0.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 40.0, 0.0], [0.0, 60.0, 0.0], [0.0, 80.0, 0.0], [0.0, 100.0, 0.0], [0.0, 120.0, 0.0], [0.0, 140.0, 0.0], [0.0, 160.0, 0.0], [0.0, 180.0, 0.0], [140.0, 0.0, 20.0], [140.0, 20.0, 20.0], [140.0, 40.0, 20.0], [140.0, 60.0, 20.0], [140.0, 80.0, 20.0], [140.0, 100.0, 20.0], [140.0, 120.0, 20.0], [140.0, 140.0, 20.0], [140.0, 160.0, 20.0], [140.0, 180.0, 20.0], [140.0, 0.0, 40.0], [140.0, 20.0, 40.0], [140.0, 40.0, 40.0], [140.0, 60.0, 40.0], [140.0, 80.0, 40.0], [140.0, 100.0, 40.0], [140.0, 120.0, 40.0], [140.0, 140.0, 40.0], [140.0, 160.0, 40.0], [140.0, 180.0, 40.0], [140.0, 0.0, 60.0], [140.0, 20.0, 60.0], [140.0, 40.0, 60.0], [140.0, 60.0, 60.0], [140.0, 80.0, 60.0], [140.0, 100.0, 60.0], [140.0, 120.0, 60.0], [140.0, 140.0, 60.0], [140.0, 160.0, 60.0], [140.0, 180.0, 60.0], [140.0, 0.0, 80.0], [140.0, 20.0, 80.0], [140.0, 40.0, 80.0], [140.0, 60.0, 80.0], [140.0, 80.0, 80.0], [140.0, 100.0, 80.0], [140.0, 120.0, 80.0], [140.0, 140.0, 80.0], [140.0, 160.0, 80.0], [140.0, 180.0, 80.0], [140.0, 0.0, 100.0], [140.0, 20.0, 100.0], [140.0, 40.0, 100.0], [140.0, 60.0, 100.0], [140.0, 80.0, 100.0], [140.0, 100.0, 100.0], [140.0, 120.0, 100.0], [140.0, 140.0, 100.0], [140.0, 160.0, 100.0], [140.0, 180.0, 100.0], [140.0, 0.0, 120.0], [140.0, 20.0, 120.0], [140.0, 40.0, 120.0], [140.0, 60.0, 120.0], [140.0, 80.0, 120.0], [140.0, 100.0, 120.0], [140.0, 120.0, 120.0], [140.0, 140.0, 120.0], [140.0, 160.0, 120.0], [140.0, 180.0, 120.0], [140.0, 0.0, 140.0], [140.0, 20.0, 140.0], [140.0, 40.0, 140.0], [140.0, 60.0, 140.0], [140.0, 80.0, 140.0], [140.0, 100.0, 140.0], [140.0, 120.0, 140.0], [140.0, 140.0, 140.0], [140.0, 160.0, 140.0], [140.0, 180.0, 140.0]]      
    
    up = [[2179.0, 872.0], [2163.0, 830.0], [2146.0, 790.0], [2132.0, 749.0], [2116.0, 709.0], [2101.0, 671.0], [2086.0, 633.0], [2072.0, 596.0], [2058.0, 558.0], [2044.0, 522.0], [2129.0, 856.0], [2113.0, 815.0], [2098.0, 774.0], [2082.0, 732.0], [2068.0, 693.0], [2053.0, 654.0], [2039.0, 616.0], [2024.0, 578.0], [2011.0, 541.0], [1996.0, 504.0], [2079.0, 840.0], [2063.0, 799.0], [2047.0, 758.0], [2033.0, 716.0], [2018.0, 677.0], [2004.0, 637.0], [1989.0, 599.0], [1976.0, 561.0], [1963.0, 522.0], [1949.0, 486.0], [2027.0, 824.0], [2011.0, 783.0], [1997.0, 741.0], [1982.0, 699.0], [1968.0, 660.0], [1953.0, 620.0], [1940.0, 582.0], [1926.0, 543.0], [1913.0, 505.0], [1900.0, 468.0], [1975.0, 807.0], [1959.0, 765.0], [1945.0, 724.0], [1930.0, 682.0], [1916.0, 642.0], [1902.0, 602.0], [1889.0, 564.0], [1876.0, 525.0], [1863.0, 486.0], [1851.0, 450.0], [1921.0, 790.0], [1906.0, 748.0], [1892.0, 706.0], [1878.0, 665.0], [1864.0, 624.0], [1851.0, 584.0], [1838.0, 545.0], [1825.0, 507.0], [1812.0, 469.0], [1800.0, 430.0], [1867.0, 772.0], [1852.0, 730.0], [1838.0, 688.0], [1824.0, 646.0], [1811.0, 606.0], [1798.0, 566.0], [1785.0, 525.0], [1773.0, 488.0], [1760.0, 450.0], [1749.0, 412.0], [2262.0, 842.0], [2244.0, 801.0], [2228.0, 760.0], [2211.0, 721.0], [2195.0, 681.0], [2179.0, 643.0], [2164.0, 604.0], [2149.0, 568.0], [2135.0, 531.0], [2121.0, 496.0], [2296.0, 799.0], [2278.0, 756.0], [2260.0, 717.0], [2243.0, 677.0], [2227.0, 637.0], [2212.0, 598.0], [2195.0, 561.0], [2181.0, 524.0], [2166.0, 488.0], [2152.0, 452.0], [2330.0, 754.0], [2311.0, 712.0], [2293.0, 672.0], [2276.0, 631.0], [2260.0, 592.0], [2243.0, 554.0], [2228.0, 516.0], [2212.0, 480.0], [2197.0, 444.0], [2182.0, 408.0], [2363.0, 707.0], [2345.0, 666.0], [2327.0, 626.0], [2310.0, 585.0], [2293.0, 547.0], [2276.0, 509.0], [2260.0, 472.0], [2245.0, 435.0], [2229.0, 398.0], [2214.0, 364.0], [2398.0, 661.0], [2380.0, 619.0], [2361.0, 579.0], [2344.0, 539.0], [2327.0, 501.0], [2310.0, 463.0], [2293.0, 425.0], [2277.0, 390.0], [2261.0, 353.0], [2246.0, 318.0], [2435.0, 613.0], [2415.0, 572.0], [2396.0, 532.0], [2379.0, 492.0], [2361.0, 454.0], [2344.0, 416.0], [2327.0, 379.0], [2311.0, 343.0], [2295.0, 307.0], [2280.0, 273.0], [2473.0, 565.0], [2452.0, 524.0], [2433.0, 484.0], [2415.0, 445.0], [2396.0, 407.0], [2379.0, 369.0], [2362.0, 332.0], [2345.0, 296.0], [2329.0, 261.0], [2313.0, 227.0]]
    down = [[2779.0, 2030.0], [2818.0, 2090.0], [2856.0, 2152.0], [2896.0, 2215.0], [2937.0, 2279.0], [2978.0, 2346.0], [3021.0, 2413.0], [3065.0, 2482.0], [3110.0, 2553.0], [3156.0, 2626.0], [2726.0, 2030.0], [2764.0, 2091.0], [2803.0, 2154.0], [2843.0, 2217.0], [2884.0, 2283.0], [2926.0, 2350.0], [2968.0, 2418.0], [3011.0, 2489.0], [3057.0, 2560.0], [3103.0, 2634.0], [2672.0, 2030.0], [2709.0, 2093.0], [2748.0, 2156.0], [2788.0, 2220.0], [2829.0, 2286.0], [2871.0, 2354.0], [2913.0, 2424.0], [2956.0, 2496.0], [3003.0, 2567.0], [3048.0, 2643.0], [2616.0, 2029.0], [2653.0, 2093.0], [2692.0, 2157.0], [2732.0, 2222.0], [2772.0, 2290.0], [2814.0, 2359.0], [2856.0, 2430.0], [2900.0, 2502.0], [2945.0, 2576.0], [2992.0, 2651.0], [2559.0, 2029.0], [2597.0, 2093.0], [2635.0, 2159.0], [2674.0, 2225.0], [2714.0, 2294.0], [2756.0, 2364.0], [2798.0, 2436.0], [2842.0, 2509.0], [2887.0, 2584.0], [2933.0, 2661.0], [2501.0, 2028.0], [2537.0, 2094.0], [2575.0, 2160.0], [2614.0, 2228.0], [2654.0, 2297.0], [2696.0, 2369.0], [2739.0, 2441.0], [2782.0, 2516.0], [2826.0, 2592.0], [2873.0, 2670.0], [2441.0, 2027.0], [2477.0, 2094.0], [2515.0, 2161.0], [2554.0, 2230.0], [2593.0, 2301.0], [2634.0, 2373.0], [2677.0, 2446.0], [2720.0, 2523.0], [2764.0, 2601.0], [2811.0, 2680.0], [2876.0, 1978.0], [2914.0, 2037.0], [2954.0, 2098.0], [2994.0, 2161.0], [3036.0, 2224.0], [3079.0, 2289.0], [3122.0, 2356.0], [3168.0, 2424.0], [3215.0, 2494.0], [3263.0, 2566.0], [2921.0, 1927.0], [2960.0, 1986.0], [3000.0, 2047.0], [3042.0, 2109.0], [3085.0, 2173.0], [3129.0, 2238.0], [3174.0, 2305.0], [3220.0, 2374.0], [3268.0, 2444.0], [3317.0, 2516.0], [2966.0, 1875.0], [3006.0, 1934.0], [3047.0, 1995.0], [3090.0, 2057.0], [3134.0, 2121.0], [3179.0, 2186.0], [3226.0, 2253.0], [3274.0, 2322.0], [3322.0, 2392.0], [3373.0, 2464.0], [3011.0, 1821.0], [3053.0, 1881.0], [3096.0, 1942.0], [3140.0, 2004.0], [3185.0, 2068.0], [3231.0, 2133.0], [3279.0, 2200.0], [3328.0, 2269.0], [3378.0, 2339.0], [3430.0, 2411.0], [3058.0, 1767.0], [3101.0, 1827.0], [3145.0, 1888.0], [3190.0, 1950.0], [3237.0, 2014.0], [3284.0, 2079.0], [3333.0, 2146.0], [3384.0, 2215.0], [3435.0, 2285.0], [3489.0, 2357.0], [3107.0, 1712.0], [3150.0, 1771.0], [3196.0, 1832.0], [3242.0, 1894.0], [3290.0, 1958.0], [3339.0, 2023.0], [3389.0, 2090.0], [3441.0, 2159.0], [3494.0, 2230.0], [3549.0, 2302.0], [3157.0, 1655.0], [3202.0, 1714.0], [3248.0, 1775.0], [3295.0, 1838.0], [3344.0, 1901.0], [3395.0, 1967.0], [3446.0, 2034.0], [3499.0, 2102.0], [3554.0, 2173.0], [3611.0, 2245.0]]
    
    up = [[[349, 131], [387, 142], [425, 155], [462, 167], [503, 180], [543, 193], [583, 206], [625, 220], [667, 234], [708, 249]], [[367, 182], [406, 194], [444, 207], [482, 220], [521, 233], [561, 246], [602, 260], [643, 274], [685, 288], [727, 303]], [[387, 233], [423, 245], [462, 258], [501, 271], [540, 284], [579, 298], [619, 312], [661, 327], [702, 341], [744, 357]], [[405, 282], [442, 295], [480, 308], [519, 322], [557, 335], [597, 350], [636, 364], [678, 379], [720, 393], [761, 409]], [[423, 331], [459, 345], [498, 358], [536, 371], [574, 386], [613, 400], [653, 415], [695, 429], [736, 445], [777, 460]], [[441, 378], [478, 393], [515, 406], [553, 420], [591, 435], [630, 450], [669, 464], [711, 480], [752, 495], [793, 511]], [[459, 426], [495, 440], [532, 454], [570, 468], [608, 483], [646, 498], [686, 514], [727, 528], [767, 545], [809, 561]], [[164, 695], [198, 711], [233, 727], [269, 744], [306, 761], [344, 778], [382, 797], [421, 815], [461, 834], [502, 855]], [[210, 662], [244, 677], [280, 693], [316, 709], [353, 726], [391, 743], [429, 761], [469, 778], [509, 797], [550, 817]], [[255, 628], [290, 643], [327, 659], [362, 675], [400, 692], [438, 709], [476, 726], [516, 743], [556, 762], [598, 780]], [[300, 596], [336, 611], [372, 627], [409, 642], [446, 658], [484, 675], [522, 692], [563, 709], [603, 727], [644, 745]], [[345, 564], [381, 579], [417, 594], [453, 610], [491, 625], [529, 642], [568, 658], [609, 675], [649, 693], [691, 712]], [[389, 534], [425, 548], [461, 563], [498, 577], [535, 594], [574, 609], [614, 625], [654, 642], [693, 660], [736, 678]], [[433, 503], [468, 517], [505, 531], [541, 546], [580, 561], [618, 577], [658, 593], [697, 610], [738, 626], [779, 644]]]
    up_real = [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0], [0.0, 7.0, 0.0], [0.0, 8.0, 0.0], [0.0, 9.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.0], [1.0, 5.0, 0.0], [1.0, 6.0, 0.0], [1.0, 7.0, 0.0], [1.0, 8.0, 0.0], [1.0, 9.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [2.0, 5.0, 0.0], [2.0, 6.0, 0.0], [2.0, 7.0, 0.0], [2.0, 8.0, 0.0], [2.0, 9.0, 0.0]], [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [3.0, 6.0, 0.0], [3.0, 7.0, 0.0], [3.0, 8.0, 0.0], [3.0, 9.0, 0.0]], [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 5.0, 0.0], [4.0, 6.0, 0.0], [4.0, 7.0, 0.0], [4.0, 8.0, 0.0], [4.0, 9.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0], [5.0, 4.0, 0.0], [5.0, 5.0, 0.0], [5.0, 6.0, 0.0], [5.0, 7.0, 0.0], [5.0, 8.0, 0.0], [5.0, 9.0, 0.0]], [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, 2.0, 0.0], [6.0, 3.0, 0.0], [6.0, 4.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [6.0, 7.0, 0.0], [6.0, 8.0, 0.0], [6.0, 9.0, 0.0]], [[7.0, 0.0, 7.0], [7.0, 1.0, 7.0], [7.0, 2.0, 7.0], [7.0, 3.0, 7.0], [7.0, 4.0, 7.0], [7.0, 5.0, 7.0], [7.0, 6.0, 7.0], [7.0, 7.0, 7.0], [7.0, 8.0, 7.0], [7.0, 9.0, 7.0]], [[7.0, 0.0, 6.0], [7.0, 1.0, 6.0], [7.0, 2.0, 6.0], [7.0, 3.0, 6.0], [7.0, 4.0, 6.0], [7.0, 5.0, 6.0], [7.0, 6.0, 6.0], [7.0, 7.0, 6.0], [7.0, 8.0, 6.0], [7.0, 9.0, 6.0]], [[7.0, 0.0, 5.0], [7.0, 1.0, 5.0], [7.0, 2.0, 5.0], [7.0, 3.0, 5.0], [7.0, 4.0, 5.0], [7.0, 5.0, 5.0], [7.0, 6.0, 5.0], [7.0, 7.0, 5.0], [7.0, 8.0, 5.0], [7.0, 9.0, 5.0]], [[7.0, 0.0, 4.0], [7.0, 1.0, 4.0], [7.0, 2.0, 4.0], [7.0, 3.0, 4.0], [7.0, 4.0, 4.0], [7.0, 5.0, 4.0], [7.0, 6.0, 4.0], [7.0, 7.0, 4.0], [7.0, 8.0, 4.0], [7.0, 9.0, 4.0]], [[7.0, 0.0, 3.0], [7.0, 1.0, 3.0], [7.0, 2.0, 3.0], [7.0, 3.0, 3.0], [7.0, 4.0, 3.0], [7.0, 5.0, 3.0], [7.0, 6.0, 3.0], [7.0, 7.0, 3.0], [7.0, 8.0, 3.0], [7.0, 9.0, 3.0]], [[7.0, 0.0, 2.0], [7.0, 1.0, 2.0], [7.0, 2.0, 2.0], [7.0, 3.0, 2.0], [7.0, 4.0, 2.0], [7.0, 5.0, 2.0], [7.0, 6.0, 2.0], [7.0, 7.0, 2.0], [7.0, 8.0, 2.0], [7.0, 9.0, 2.0]], [[7.0, 0.0, 1.0], [7.0, 1.0, 1.0], [7.0, 2.0, 1.0], [7.0, 3.0, 1.0], [7.0, 4.0, 1.0], [7.0, 5.0, 1.0], [7.0, 6.0, 1.0], [7.0, 7.0, 1.0], [7.0, 8.0, 1.0], [7.0, 9.0, 1.0]]]
    
    down = [[[602, 200], [669, 236], [736, 274], [805, 313], [876, 352], [948, 393], [1021, 436], [1098, 479], [1176, 523], [1255, 570]], [[603, 260], [669, 296], [735, 334], [803, 373], [872, 413], [944, 455], [1016, 498], [1091, 541], [1168, 585], [1245, 632]], [[604, 318], [668, 356], [734, 394], [800, 433], [869, 473], [939, 515], [1011, 557], [1084, 601], [1159, 646], [1236, 692]], [[604, 376], [668, 412], [732, 451], [797, 491], [865, 531], [934, 573], [1005, 615], [1077, 659], [1151, 704], [1226, 751]], [[605, 431], [667, 468], [731, 507], [795, 547], [861, 588], [929, 630], [999, 672], [1071, 715], [1142, 762], [1218, 807]], [[605, 485], [666, 523], [729, 562], [792, 602], [858, 643], [925, 685], [993, 727], [1064, 770], [1135, 816], [1209, 862]], [[605, 538], [665, 577], [727, 615], [790, 655], [854, 696], [921, 738], [988, 780], [1057, 824], [1128, 869], [1201, 915]], [[230, 916], [289, 961], [350, 1007], [413, 1054], [476, 1103], [542, 1154], [609, 1205], [677, 1258], [748, 1313], [820, 1370]], [[287, 866], [346, 909], [407, 955], [469, 1001], [533, 1049], [598, 1098], [665, 1148], [734, 1200], [805, 1253], [877, 1308]], [[342, 817], [402, 860], [463, 904], [525, 949], [589, 996], [654, 1043], [721, 1092], [790, 1143], [860, 1194], [932, 1248]], [[396, 770], [456, 812], [517, 855], [579, 899], [643, 944], [708, 990], [775, 1038], [844, 1087], [914, 1137], [986, 1189]], [[450, 725], [509, 765], [570, 806], [632, 849], [696, 893], [761, 938], [828, 985], [897, 1033], [967, 1081], [1039, 1132]], [[502, 680], [561, 719], [622, 759], [684, 801], [748, 844], [813, 888], [880, 933], [949, 979], [1019, 1027], [1091, 1076]], [[553, 635], [612, 673], [673, 713], [736, 753], [799, 795], [864, 838], [931, 881], [999, 927], [1069, 974], [1141, 1022]]]
    down_real = [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0], [0.0, 7.0, 0.0], [0.0, 8.0, 0.0], [0.0, 9.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [1.0, 4.0, 0.0], [1.0, 5.0, 0.0], [1.0, 6.0, 0.0], [1.0, 7.0, 0.0], [1.0, 8.0, 0.0], [1.0, 9.0, 0.0]], [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [2.0, 5.0, 0.0], [2.0, 6.0, 0.0], [2.0, 7.0, 0.0], [2.0, 8.0, 0.0], [2.0, 9.0, 0.0]], [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0], [3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [3.0, 6.0, 0.0], [3.0, 7.0, 0.0], [3.0, 8.0, 0.0], [3.0, 9.0, 0.0]], [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 5.0, 0.0], [4.0, 6.0, 0.0], [4.0, 7.0, 0.0], [4.0, 8.0, 0.0], [4.0, 9.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0], [5.0, 4.0, 0.0], [5.0, 5.0, 0.0], [5.0, 6.0, 0.0], [5.0, 7.0, 0.0], [5.0, 8.0, 0.0], [5.0, 9.0, 0.0]], [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, 2.0, 0.0], [6.0, 3.0, 0.0], [6.0, 4.0, 0.0], [6.0, 5.0, 0.0], [6.0, 6.0, 0.0], [6.0, 7.0, 0.0], [6.0, 8.0, 0.0], [6.0, 9.0, 0.0]], [[7.0, 0.0, 7.0], [7.0, 1.0, 7.0], [7.0, 2.0, 7.0], [7.0, 3.0, 7.0], [7.0, 4.0, 7.0], [7.0, 5.0, 7.0], [7.0, 6.0, 7.0], [7.0, 7.0, 7.0], [7.0, 8.0, 7.0], [7.0, 9.0, 7.0]], [[7.0, 0.0, 6.0], [7.0, 1.0, 6.0], [7.0, 2.0, 6.0], [7.0, 3.0, 6.0], [7.0, 4.0, 6.0], [7.0, 5.0, 6.0], [7.0, 6.0, 6.0], [7.0, 7.0, 6.0], [7.0, 8.0, 6.0], [7.0, 9.0, 6.0]], [[7.0, 0.0, 5.0], [7.0, 1.0, 5.0], [7.0, 2.0, 5.0], [7.0, 3.0, 5.0], [7.0, 4.0, 5.0], [7.0, 5.0, 5.0], [7.0, 6.0, 5.0], [7.0, 7.0, 5.0], [7.0, 8.0, 5.0], [7.0, 9.0, 5.0]], [[7.0, 0.0, 4.0], [7.0, 1.0, 4.0], [7.0, 2.0, 4.0], [7.0, 3.0, 4.0], [7.0, 4.0, 4.0], [7.0, 5.0, 4.0], [7.0, 6.0, 4.0], [7.0, 7.0, 4.0], [7.0, 8.0, 4.0], [7.0, 9.0, 4.0]], [[7.0, 0.0, 3.0], [7.0, 1.0, 3.0], [7.0, 2.0, 3.0], [7.0, 3.0, 3.0], [7.0, 4.0, 3.0], [7.0, 5.0, 3.0], [7.0, 6.0, 3.0], [7.0, 7.0, 3.0], [7.0, 8.0, 3.0], [7.0, 9.0, 3.0]], [[7.0, 0.0, 2.0], [7.0, 1.0, 2.0], [7.0, 2.0, 2.0], [7.0, 3.0, 2.0], [7.0, 4.0, 2.0], [7.0, 5.0, 2.0], [7.0, 6.0, 2.0], [7.0, 7.0, 2.0], [7.0, 8.0, 2.0], [7.0, 9.0, 2.0]], [[7.0, 0.0, 1.0], [7.0, 1.0, 1.0], [7.0, 2.0, 1.0], [7.0, 3.0, 1.0], [7.0, 4.0, 1.0], [7.0, 5.0, 1.0], [7.0, 6.0, 1.0], [7.0, 7.0, 1.0], [7.0, 8.0, 1.0], [7.0, 9.0, 1.0]]]

    offsetUp = (62,1617)
    offsetDown = (1424 , 2240)
    
    shape = (3072,4608)
    
    o.calibrateStereo(filenames, shape, up, down, up_real, down_real, offsetUp, offsetDown)
    
    