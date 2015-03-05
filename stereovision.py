  #!/usr/bin/python -u
# -*- coding: utf-8 -*-
'''
Created on Feb 26, 2015

@author: Tomasz
'''
import sys, os
import cv2
import time
import gc
 
import numpy as np
import func.markElements as mark
from scene.scene import Scene
from scene.qubic import QubicObject

from scene.mirrorDetector import mirrorDetector
from scene.objectDetector2 import objectDetector2

from geoobjects import Recovery

import calculations.calibration3 as calibration
from calculations.chessboard import ChessboardDetector

GENERATE_RESULT = True
results_folder = 'results'

writepath = results_folder + '/'
calibration.writepath = writepath

# adjustable parameters for different types of scene

chessboardDetectionTreshold = [150,150]  # -> empirical for the background of the chesssboard
cornersVerificationCircleDiameter = [9, 31]  # -> usually less than 25% of chessboard field pixel size o: mirrored, direct
board_w = 10
board_h = 7

thresholdGetFields = [120,120,120,120,120]


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
    print len(points)
    print len(points[0])
    ps = points
    shape = (14,10)
    
    mirror = np.zeros((shape[0],shape[1],2))
    for x in range(shape[0]):
        for y in range(shape[1]):
            mirror[x][shape[1] - y - 1] = ps[x][y]
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


def swap(A):
    A = np.array(A)
    
    A = A.T
    B = np.array([ A[1],A[0] ])
    B = B.T
    
    return B

def rearange(A):
    print 'shape',A.shape
    D = A.shape[2]
    rows = 14
    cols = 10
    print 'destination shape',(rows,cols,D)
    A.reshape((rows,cols,D))
    C = np.zeros_like(A)
    
    for row in range(rows):
        C[row] = A [ ((row/7)*7 + 6-(row%7)) ]
    
    B = C.tolist()
    
    return B
    
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

   
def drawAxes(img, imgpts):
#     corner = tuple(origin.ravel())
    o2 = (int (imgpts[3][0][1]) ,int (imgpts[3][0][0]))
    corner = o2
    
    cv2.line(img, (corner[1],corner[0]), (int (imgpts[0][0][1]) ,int (imgpts[0][0][0])), (255,0,0), 5)
    cv2.line(img, (corner[1],corner[0]), (int (imgpts[1][0][1]) ,int (imgpts[1][0][0])), (0,255,0), 5)
    cv2.line(img, (corner[1],corner[0]), (int (imgpts[2][0][1]) ,int (imgpts[2][0][0])), (0,0,255), 5)
    
    cv2.circle(img,o2,5,(255,255,255),-1)
#         cv2.line(img, (corner[1],corner[0]), (int (imgpts[3][0][1]) ,int (imgpts[3][0][0])), (255,255,255), 5)
#         
#         cv2.circle(img,(int (imgpts[4][0][1]) ,int (imgpts[4][0][0])),2,(255,0,255),-1)
#         cv2.circle(img,(int (imgpts[5][0][1]) ,int (imgpts[5][0][0])),2,(255,0,255),-1)
#         cv2.circle(img,(int (imgpts[6][0][1]) ,int (imgpts[6][0][0])),2,(255,0,255),-1)
#         cv2.circle(img,(int (imgpts[7][0][1]) ,int (imgpts[7][0][0])),2,(255,0,255),-1)
    return img

def drawBundarites(img,points):
    for corner in points:
        corner = tuple(corner.ravel())
        corner = map(int,corner)
        print (corner[1],corner[0])
        cv2.circle(img, (corner[1],corner[0]), 55 , (255,0,255),  -1)
    return img
    
    
def makeWow(img1, mtx, dist, rvecs, tvecs):
    axis = np.float32([[40,0,0], [0,40,0], [0,0,40],[0,0,0]]).reshape(-1,3)
    imgpoints,jacobian = cv2.projectPoints(axis, rvecs,tvecs, mtx, dist)
    img1 = drawAxes(img1, imgpoints)
    
    originpoints,jacobian = cv2.projectPoints(np.float32([[0,0,0]]).reshape(-1,3), rvecs,tvecs, mtx, dist) 
    print 'wow'
    
    points = np.float32([[120,180,0], [0,0,0], [140,0,140],[140,180,20]]).reshape(-1,3)
    
    points,jacobian = cv2.projectPoints(points, rvecs,tvecs, mtx, dist) 
    
    img1 = drawBundarites(img1, points)
    
    

   
def calibrate():
    # nastepuje kalibracja kamery obrazami szachownicy
    names1 = sys.argv[2]
    names2 = sys.argv[3]
    names3 = sys.argv[4]
    if names1 == 'None':
        mirrored = []
    else:
        mirrored = names1.split("|")
        
    if names2 == 'None':
        direct = []
    else:
        direct = names2.split("|")
    if names3 == 'None':
        both = []
    else:
        both = names3.split("|")
        
    for index,b in enumerate(both):
        print b
        scene = loadImage(b)
        md = mirrorDetector(scene)
        mirrorZone = md.findMirrorZone()
        filenames = calibration.prepareCalibration(md, index+1)
        mirrored.append(filenames[0])
        direct.append(filenames[1])
        
    counter = len(mirrored)
    print 'counter', counter
    mirrored.extend(direct)
    
    print 'calibration files:', mirrored
    
    imag = []
    real = []
    
    outputImages = []
    filenames = []
    offsets = []
    
    i = 0
    for index, filename in enumerate(mirrored):
        if index>=counter:
            print 'zmiana'
            i = 1
            
        print '=====' 
        print 'calibrate image', filename
        print '====='
        
        scene = loadImage(filename)
        
        shape = (scene.view.shape[0],scene.view.shape[1])
#         outputImages.append(scene.view)
        filenames.append(filename)

        cd = ChessboardDetector(chessboardDetectionTreshold[i], (board_w, board_h), cornersVerificationCircleDiameter[i])
        cd.image_index = index
        cd.image_type = i
        parts = filename.split('/')
        cd.filename = parts[-1]
        print 'FILE:',cd.filename 
        
        cd.thresholdGetFields = thresholdGetFields[index]
        
        finalP, finalWP, image, offset = calibration.getCalibrationPointsForScene(scene, cd)
        
        if i == 0:
            finalP = np.array(MirrorPoints(finalP))
        else:
            finalP = np.array(finalP)
            
        calibration.saveParameter(finalP, 'ip_%s' % filename.split('/')[-1])
        calibration.saveParameter(finalWP, 'rp_%s' % filename.split('/')[-1])
        
        imag.append(finalP)
        real.append(finalWP)  
        offsets.append(offset)
        gc.collect()
    
    chessPoints = []
    for im in imag:
        p = rearange(im)
        chessPoints.append(p)
        gc.collect()
    print 'real'
    real = rearange(real[0])
    
    trueOffsets = []
    for off in offsets:
        offset = off[1],off[0]
        trueOffsets.append( offset )

    calibrationTool = Recovery.thirdDianensionREcovery()
    imagePoints2, mtx, dist, rvecs, tvecs, left_real, images = calibrationTool.calibrateMulti(filenames, shape, chessPoints, real, trueOffsets)
    
#     for rvec,tvec,imgPoints, img1,filename in zip(rvecs,tvecs,imagePoints2,outputImages,filenames):
#         f = 'results/calibration/calibrated_image_%s' % filename
#         print 'results/calibration/calibrated_image_%s' % filename
#         cv2.imwrite(f, img1)
        
    print mtx
    print dist
    print 'R', rvecs[-1]
    print 'T', tvecs[-1]
    
    for image,filename in zip(images,filenames):
        fnames = filename.split('/')
        cv2.imwrite('results/calibration/source/'+fnames[-1],image)
    
    calibration.saveIntrinsicCalibration(mtx,dist)


def poseEstimation():
    filename = sys.argv[2]
    scene = loadImage(filename)
    shape = scene.view.shape[:2]
    print 'shape = ',shape
    md = mirrorDetector(scene)
    mirrorZone = md.findMirrorZone()
    
    mtx,dist = calibration.loadIntrinsicCalibration()
    mtx0  = mtx
    dist0 = dist
    
    index = 0
    filenames = calibration.prepareCalibration(md, index)
    print 'filenames = ',filenames
    
    imag = {}
    real = {}
    
    board_w = 10
    board_h = 7
    
    views = []
    offsets = []
    
    for i,filename in enumerate(filenames):
        scene = loadImage(filename)
        parts = filename.split('/')
        
        cd = ChessboardDetector(chessboardDetectionTreshold[i], (board_w, board_h), cornersVerificationCircleDiameter[i])
        cd.image_index = index
        cd.image_type = i
        cd.filename = parts[-1]
        print 'FILE:',cd.filename 
        cd.thresholdGetFields = 130
        
        finalP, finalWP, image, offset = calibration.getCalibrationPointsForScene(scene, cd)
        
        offsets.append(offset)
        
        if i == 0:
            finalP = np.array(MirrorPoints(finalP))
            pass
        else:
            finalP = np.array(finalP)
            
        views.append(cd.origin.copy())
        
        imag[i] = finalP
        real[i] = finalWP  
        
#         window = cv2.namedWindow('a',cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('a',shape[0]/4,shape[1]/4)
#         cv2.imshow('a',scan)
#     #     time.sleep(10)
#         cv2.waitKey()
#         cv2.destroyAllWindows()

    calibrationTool = Recovery.thirdDianensionREcovery()
    
    trueOffsets = []
    for off in offsets:
        offset = off[1],off[0]
        trueOffsets.append( offset )
    
    up = imag[0]
    up = rearange(up)
    
    down = imag[1]
    down = rearange(down)

    real[0] = rearange(real[0])
    
#     modelPoints = [
#                    [(431,1174),(479,1311),(385,1290),(338,1156)],
#                    [(1877,1635),(1880,1804),(2037,1872),(2038,1701)]
#                    ]
    
    imagePoints2, mtx, dist, rvecs, tvecs, real, images = calibrationTool.calibrateMulti(filenames, shape, [up, down], real[0], trueOffsets, True, mtx0 ,dist0)
   
    
    P1,P2,F = calibrationTool.getFundamental(imagePoints2, rvecs, tvecs, mtx, dist, real, images, modelPoints=None)
    print P1
    print P2
    print F
    calibration.saveParameter(P1, 'P1')
    calibration.saveParameter(P2, 'P2')
    calibration.saveParameter(F, 'Fundamental')
    calibration.saveParameter(np.array(filenames), 'filenames')
    
    calibration.saveParameter(np.array(trueOffsets), 'TrueOffset')
    
    calibration.saveParameter(np.array(imagePoints2),'imagePoints2')

        
def localize():
    d = calibration.loadParameter('TrueOffset')
    offsets = d[0].tolist()

    #wczytanie parametrów
    filenames = calibration.loadParameter('filenames')
    P1 = calibration.loadParameter('P1')
    P2 = calibration.loadParameter('P2')
    F = calibration.loadParameter('Fundamental')
    imp = calibration.loadParameter('imagePoints2')
    
    filenames = filenames[0].tolist()
    scenes = []
    for f in filenames:
        scenes.append( loadImage(f) )
    images = []
    for scene in scenes:
        images.append( scene.view.copy() )
            
    fundamental = F[0]
    P1 = P1[0]
    P2 = P2[0]
    imagePoints2 = imp[0]
    
    filename = sys.argv[2]
    print filename
    
    # nastepuje lokalizacja obiektu
    scene = loadImage(filename)
    md = mirrorDetector(scene)
    mirrorZone = md.findMirrorZone()
    
    f = writepath + 'mirror_zone.jpg'
    print 'zapisywanie do ' + f
    cv2.imwrite(f, mirrorZone.preview)
    
    gray = scene.gray
    dst = cv2.cornerHarris(gray,blockSize=5,ksize=3,k=0.04)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),1,cv2.THRESH_BINARY)
    indieces =  np.nonzero(dst)
    corners = np.array([indieces[0],indieces[1]]).T
    dst2 = np.zeros_like(gray)
    ct = corners.T
    dst2[(ct[0],ct[1])] = 1
    bb = np.zeros((gray.shape[0],gray.shape[1],3))
    bb = scene.view.copy()
    bb[dst2>0] = (255,0,0)
    cv2.imwrite('results/Harris_preview.jpg' , bb)
    
    od = objectDetector2(md, md.origin)
    zoneA, zoneC = od.detect(chessboard=False, multi=False)
    
    f = writepath + 'objectA.jpg'
    print 'zapisywanie do' + f
    cv2.imwrite(f, zoneA.image)
    
    f = writepath + 'objectC.jpg'
    print 'zapisywanie do ' + f
    cv2.imwrite(f, zoneC.image)
    
    modelPoints= [[],[]]
    
    qubicA = QubicObject(zoneA.image)
    
    
    cv2.imwrite('results/sobelA.jpg',np.where(qubicA.edgeMask>0,255,0)) 
    
    imgQ = mark.drawQubic(qubicA)
    cv2.imwrite('results/object_structure_A.jpg',imgQ)
    
    qubicC = QubicObject(zoneC.image)
    
    cv2.imwrite('results/sobelC.jpg',np.where(qubicC.edgeMask>0,255,0)) 
    mark.drawQubic(qubicC)
    imgQ = mark.drawQubic(qubicC)
    cv2.imwrite('results/object_structure_C.jpg',imgQ) 
    
    objectsOffsets = [[],[]]
     
    modelPoints[0] = qubicA.getTopWall()
    objectsOffsets[0] = (zoneA.offsetX, zoneA.offsetY)
    
    modelPoints[1] = qubicC.getTopWall()
    objectsOffsets[1] = (zoneC.offsetX, zoneC.offsetY)
    
#     print objectsOffsets
    
    modelPoints2 = []
    for points,offset in zip(modelPoints,objectsOffsets):
        tmp =  offset + np.array(points)
        modelPoints2.append( tmp )
    
    #rysowanie puntow
    d = 0
    for i,model in zip(images,modelPoints2):
        img = scene.view.copy() 
        for p in model:
            cv2.circle(img, (p[0],p[1]), 5, (0,0,0), -1) 
        cv2.imwrite('results/marked%d.jpg'%d,img)
        d += 1
    
    #pasowanie punktow
    oPoints1 = np.array(modelPoints2[0],dtype='float32')
    oPoints2 = np.array(modelPoints2[1],dtype='float32')
    
    calibrationTool = Recovery.thirdDianensionREcovery()
    
    oPoints1 = swap(oPoints1)
    oPoints2 = swap(oPoints2)
    
    lines1, lines2 = calibrationTool.calculateEpilines(oPoints1, oPoints2, fundamental, scene.view.copy(), scene.view.copy())
    
    oPoints1, oPoints2 = calibrationTool.matchPoints(oPoints1,oPoints2,lines1,lines2,scene.view.copy(), scene.view.copy())
    
    #triangulacja
    imagePoints3 = np.append(imagePoints2[0], oPoints1, 0)
    imagePoints4 = np.append(imagePoints2[1], oPoints2, 0)
    
#     print imagePoints2[0]
#     print imagePoints2[1]
    
    n = imagePoints3.shape[0]
    m = oPoints1.shape[0]
    
#     imagePoints3 = imagePoints3.reshape(1,144,2)
#     imagePoints4 = imagePoints4.reshape(1,144,2)
    
#     print imagePoints3.T
    
    np.set_printoptions(precision=6,suppress=True)
    
    rrr2 = cv2.triangulatePoints(P1,P2,imagePoints3.T , imagePoints4.T)
    rrr2 = rrr2.astype(np.float32)
    
    vfunc = np.vectorize(round)
    
    
    rec =  rrr2/rrr2[3]
    
    points = cv2.convertPointsFromHomogeneous(rrr2.T)
     
    points2 = vfunc(points,4)
    print 'recovered2:\n', points2.reshape(n,3)[-m:]
    finalPoints = points2.reshape(n,3)[-m:]
    print finalPoints[0]-finalPoints[1]
    print finalPoints[1]-finalPoints[2]
    print finalPoints[2]-finalPoints[3]
    print finalPoints[3]-finalPoints[0]
    
    

if __name__ == '__main__':
    # sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    start = time.time()
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
        localize()
        
    else:
        massage = "Podaj tryb pracy i nazwy obrazów źródłowych"
        print massage
    
    end = time.time()
    print 'elapsed:', end-start


