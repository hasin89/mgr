#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import json
import os

'''
Created on Jan 27, 2015

@author: Tomasz
'''
        
writepath = ''
calibration_folder = 'calibration'

        
def getCalibrationPointsForScene(scene, chessboard_detector,chessboardFieldSize=1):
    
    corners, z = chessboard_detector.find_potential_corners(scene)

    offset = (z.offsetX, z.offsetY)
    print 'offset', offset
    corners2Shifted = chessboard_detector.getZoneCorners(corners, offset)
    
    f = 'tmp_chessboard.jpg'
    cv2.imwrite(f, z.image)
    gray = cv2.imread(f, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    chessboard_type = 0
    finalPoints, finalWorldPoints = chessboard_detector.getPoints(corners2Shifted, gray,chessboard_type)
    
    finalP = finalPoints.tolist()
    finalWP = finalWorldPoints.tolist()
    image = z.image
    
    # dodanie translacji i przemnożenie przez rozmiar pola szachownicy
    finalWP = np.multiply(finalWP, chessboardFieldSize)
#     finalP= chessboard_detector.getGlobalCorners(finalP,offset)
    
    return finalP, finalWP, image, offset


def acumulateCalibrationPoints(images,objects,(board_w,board_h), flat):
    '''
    images- punkty obrazowe zebrane z wielu obrazow
    objects - punkty rzeczywise zebrane z wielu obrazow
    flat - flaga oznaczajaca czy szachownica ma byc plaska czy przestrzenna
    '''
    
    n2 = board_w * board_h * 2
    n = board_w * board_h
    
    # płaskie punkty
    if flat == True:
        limit = n
    else:
        limit = n2
   
    objectPoints = []
    imagePoints = []
    for im, re in zip(images, objects):
        objectPoints.append(np.array(re).reshape((n2, 3))[:limit])
        imagePoints.append(im.reshape((n2, 2))[:limit])
        
    objectPoints2 = np.array(objectPoints, 'float32')
    imagePoints2 = np.array(imagePoints, 'float32')
    
    return objectPoints2, imagePoints2

def saveIntrinsicCalibration(mtx, dist):
    mlist = mtx.tolist()
    distlist = dist.tolist()
    parameters = {
                   'camera' : mlist,
                   'distortions' : distlist,
                  }
    json_string = json.dumps(parameters)
    f = getCalibrationPath()+'IntrinsicCalibration.json'
    f1 = open(f,'w')
    f1.write(json_string)
    f1.close()
    
def saveExtrinsicCalibration(rvecs, tvecs):

    rveclist = rvecs[0].tolist()
    tveclist = tvecs[0].tolist()
    parameters = {
                   'rotation' : rveclist,
                   'translation' : tveclist,
                  }
    json_string = json.dumps(parameters)
    f = getCalibrationPath()+'ExtrinsicCalibration.json'
    f1 = open(f,'w')
    f1.write(json_string)
    f1.close()
    
def saveParameter(NParray,filename):

    rveclist = NParray.tolist()
    parameters = {
                   'parameter' : rveclist,
                  }
    json_string = json.dumps(parameters)
    f = getCalibrationPath()+filename+'.json'
    f1 = open(f,'w')
    f1.write(json_string)
    f1.close()
    
def loadParameter(filename):
    f = getCalibrationPath()+filename+'.json'
    f1 = open(f,'r')
    lines = f1.readline()
    jobject = json.loads(lines)
    f1.close()
    
    return map(np.array, [jobject['parameter']])
    
    
def loadIntrinsicCalibration():
    f = getCalibrationPath()+'IntrinsicCalibration.json'
    f1 = open(f,'r')
    lines = f1.readline()
    jobject = json.loads(lines)
    f1.close()
    
    return map(np.array, [jobject['camera'],jobject['distortions']])


def loadExtrinsicCalibration():
    f = getCalibrationPath()+'ExtrinsicCalibration.json'
    f1 = open(f,'r')
    lines = f1.readline()
    jobject = json.loads(lines)
    f1.close()
    
    return map(np.array, [jobject['rotation'],jobject['translation']])


def prepareCalibration(md, index):
    '''
    utworzenie odzielnych obrazów do calów kalibracji na podstawie dwóch szachownic na jednym obrazie
    '''
    img = {}
    image = md.origin
    y = md.middle[1]
    
    
    img[1] = md.getReflectedZone()
    img[2] = md.getDirectZone()
    
    cv2.imwrite('up.jpg',img[1].image)
    cv2.imwrite('down.jpg',img[2].image)
        
    if not os.path.isdir('results'):
        print 'tworzenie folderu \'results\''
        os.mkdir('results')
    origin = os.getcwd()
    
    os.chdir('results')
    if not os.path.isdir(calibration_folder):
        print 'tworzenie folderu dla celów kalibracji'
        os.mkdir(calibration_folder)
    os.chdir(origin)
    
    files = []
    for i in range(1, 3):
        f = getCalibrationPath() + '%d_%d.jpg' % (index, i)
        print 'savaing to ' + f
        cv2.imwrite(f, img[i].preview)
        files.append(f)
        
    return files
        
def getCalibrationPath():
    if writepath == '':
        raise Exception('write path of calibration module is not set!')
    calibration_path = writepath + calibration_folder + '/'
    return calibration_path