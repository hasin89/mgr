  #!/usr/bin/python -u
# -*- coding: utf-8 -*-
'''
Created on Feb 26, 2015

@author: Tomasz
'''
import sys, os
import cv2
import json
 
import numpy as np
import func.markElements as mark
from scene.scene import Scene
from scene.zone import Zone
from scene.qubic import QubicObject

from scene.mirrorDetector import mirrorDetector
from scene.objectDetector2 import objectDetector2
from scene.edgeDetector import edgeDetector

import calculations.calibration3 as calibration
from calculations.chessboard import ChessboardDetector

GENERATE_RESULT = True
results_folder = 'results'

writepath = results_folder + '/'
calibration.writepath = writepath

# adjustable parameters for different types of scene

chessboardDetectionTreshold = 150  # -> empirical for the background of the chesssboard
cornersVerificationCircleDiameter = [10, 26]  # -> usually less than 25% of chessboard field pixel size o: mirrored, direct
board_w = 10
board_h = 7


def loadImage(filename, factor=1):
    '''
        funkcja wczytująca obrazy z plików i ewentualni je skalujaca
    '''
    
    imgT = cv2.imread(filename)
    
    if factor != 1:
        shape = (round(factor * imgT.shape[1]), round(factor * imgT.shape[0]))
        imgMap = np.empty(shape, dtype='uint8')
        imgMap = cv2.resize(imgT, imgMap.shape)
    else:
        imgMap = imgT
    
    scene = Scene(imgMap)
    return scene


def prepareResults():
    '''
    tworzy folder z wynikami
    '''
    if not os.path.isdir('results'):
        print 'tworzenie folderu \'results\''
        os.mkdir('results')
    GENERATE_RESULT = True
   
                
def detect(zone, letter):
    
    qubic = QubicObject(zone.image)
    
    # caly bialy
    image = qubic.emptyImage.copy()
    image3 = image.copy()
    
    for kk, wall in qubic.walls.iteritems():
        
        # zaznaczenie powieszhni ściany           
        image3[wall.map == 1] = (255, 255, 255) 
        for c in wall.contours:
            ll = map(np.array, np.transpose(np.array(c.points)))
            image3[ll] = (255, 255, 0)
        
#                 mark.drawHoughLines(c.lines, image3, (128,0,128), 1) 
        mark.drawHoughLines(wall.lines, image3, (128, 0, 128), 1)
            
    for vv in qubic.vertexes:
        cv2.circle(image3, (vv[0], vv[1]), 1, (10, 0, 255), 2) 
    if len(qubic.vertexes) > 7:
#             raise Exception('too much vertexes')
        pass
    print 'wierzcholki: ', qubic.vertexes
    
    f = writepath + 'lines_object_%s.jpg' % (letter)
    print 'savaing to ' + f
    cv2.imwrite(f, image3)
    
    image = zone.image.copy()
    
    for vv in qubic.vertexes:
        cv2.circle(image, (vv[0], vv[1]), 1, (10, 0, 255), 2) 
        
    f = writepath + 'pints_object_%s.jpg' % (letter)
    print 'savaing to ' + f
    cv2.imwrite(f, image)
        
    return qubic.vertexes
        

def MirrorPoints(points):
    ps = points
    mirror = np.zeros(ps.shape)
    for x in range(ps.shape[0]):
        for y in range(ps.shape[1]):
            mirror[x][ps.shape[1] - y - 1] = ps[x][y]
    return mirror      


def showDifference(filenames,imagePoints2,imagePointsR):
    
    numBoards = len(filenames)
    board_n = 140
    
    for idx in range(numBoards):
        img = cv2.imread(filenames[idx])
        for i in range(board_n): 
            cv2.circle(img,(imagePoints2[i][1],imagePoints2[i][0]),5,(0,255,0),-1)
            cv2.circle(img,(imagePointsR[i][0][1],imagePointsR[i][0][0]),6,(0,0,255),2)
        cv2.imwrite(writepath+'difference'+str(idx)+'.jpg',img)  


def calcError(imagePoints,imagePointsR):
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
            errorX += abs(imagePoints[i][0] - imagePointsR[i][0][0])
            errorY += abs(imagePoints[i][1] - imagePointsR[i][0][1])
    errorX /= numBoards * board_n
    errorY /= numBoards * board_n
    
    return (errorX,errorY)
   
   
def calibrate():
    # nastepuje kalibracja kamery obrazami szachownicy
    names1 = sys.argv[2]
    names2 = sys.argv[3]
    
    mirrored = names1.split("|")
    direct = names2.split("|")
    counter = len(mirrored)
    print 'counter', counter
    mirrored.extend(direct)
    
    imag = []
    real = []
    
    i = 0
    
    for index, filename in enumerate(mirrored):
        if index>=counter:
            i = 1
        image_index = index+1
            
        scene = loadImage(filename)
        shape = (scene.view.shape[0],scene.view.shape[1])
        
#             for i in range(0, 2):
        print '=====' 
        print 'calibrate image', filename
        print '====='
        
        scene = loadImage(filename)

        cd = ChessboardDetector(chessboardDetectionTreshold, (board_w, board_h), cornersVerificationCircleDiameter[1])
        cd.image_index = index
        cd.image_type = i
        cd.filename = filename
        
        finalP, finalWP, image, offset = calibration.getCalibrationPointsForScene(scene, cd)
        
        if i == 0:
            finalP = MirrorPoints(finalP)
        else:
            finalP = finalP
        
        imag.append(finalP)
        real.append(finalWP)  
            
    
    flat = True
    print 'acumulate'
    
    objectPoints2, imagePoints2 = calibration.acumulateCalibrationPoints(imag,real,(b_w,b_h), flat)
    
    print objectPoints2
    print imagePoints2
    print shape
    
    ret, mtx_tmp, dist_tmp, rvecs, tvecs = cv2.calibrateCamera(objectPoints2, imagePoints2, shape)
    print mtx_tmp
    print dist_tmp

    flat = False
    objectPoints2, imagePoints2 = calibration.acumulateCalibrationPoints(imag,real,(b_w,b_h), flat)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints2, imagePoints2, shape, mtx_tmp, dist_tmp, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    print mtx
    print dist
    print 'R', rvecs[-1]
    print 'T', tvecs[-1]
    
    calibration.saveIntrinsicCalibration(mtx,dist)


def poseEstimation():
    filename = sys.argv[2]
    scene = loadImage(filename)
    shape = scene.view.shape
    md = mirrorDetector(scene)
    mirrorZone = md.findMirrorZone()
    
    mtx,dist = calibration.loadIntrinsicCalibration()
    print mtx
    print dist
    
    index = 0
    filenames = calibration.prepareCalibration(md, index)
    
    imag = []
    real = []
    
    for i,filename in enumerate(filenames):
        scene = loadImage(filename)
        
        cd = ChessboardDetector(chessboardDetectionTreshold, (b_w, b_h), cornersVerificationCircleDiameter[i])
        cd.image_type = i
        cd.filename = 'pose.jpg'
        
        finalP, finalWP, image, offset = calibration.getCalibrationPointsForScene(scene, cd)
        
        if i == 0:
            finalP = MirrorPoints(finalP)
        else:
            finalP = finalP
        
        imag.append(finalP)
        real.append(finalWP)   

    flat = False
    objectPoints2, imagePoints2 = calibration.acumulateCalibrationPoints(imag,real,(b_w,b_h), flat)
    print objectPoints2
    print imagePoints2
    print objectPoints2, imagePoints2, shape[:2], mtx, dist, cv2.CALIB_USE_INTRINSIC_GUESS
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints2, imagePoints2, shape[:2], mtx, dist, flags=cv2.CALIB_USE_INTRINSIC_GUESS )
    print rvecs, tvecs
    calibration.saveExtrinsicCalibration(rvecs, tvecs)
       
        
def localize():
    filename = sys.argv[2]
    print filename
    
    #wczytanie kalibracji
    mtx, dist = calibration.loadIntrinsicCalibration()
    rvec, tvec = calibration.loadExtrinsicCalibration()
    
    print mtx
    print dist
    print 'R', rvec
    print 'T', tvec 
    
    # nastepuje lokalizacja obiektu
    scene = loadImage(filename)
    md = mirrorDetector(scene)
    mirrorZone = md.findMirrorZone()
    
    f = writepath + 'mirror_zone.jpg'
    print 'zapisywanie do ' + f
    cv2.imwrite(f, mirrorZone.preview)
    
    od = objectDetector2(md, md.origin)
    zoneA, zoneC = od.detect(chessboard=True, multi=True)
    
    f = writepath + 'objectA.jpg'
    print 'zapisywanie do' + f
    cv2.imwrite(f, zoneA.image)
    
    f = writepath + 'objectC.jpg'
    print 'zapisywanie do' + f
    cv2.imwrite(f, zoneC.image)
    
    print zoneA
    qubicA = QubicObject(zoneA.image)
    
#         ed = edgeDetector(zoneC.image.copy())
#         mask = ed.getSobel()
    
    qubicC = QubicObject(zoneC.image)
        

if __name__ == '__main__':
    # sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    scriptName = sys.argv[0]
    (b_w, b_h) = (10, 7)
    
    prepareResults()
    
    if sys.argv[1] == 'calibrate':
        print 'internal parameters calibration'
        calibrate()
        
    elif sys.argv[1] == 'pose':
        print 'pose estimation'
        poseEstimation()
        
    elif sys.argv[1] == 'localize':
        print 'localization'
        
    else:
        massage = "Podaj tryb pracy i nazwy obrazów źródłowych"
        print massage


