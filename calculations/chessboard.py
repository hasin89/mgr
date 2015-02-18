#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from calculations import labeling
from scene.zone import Zone
from scene.scene import Scene
from scene import edgeDetector, zone
from skimage import measure
from skimage import feature
from skimage import morphology,draw,transform

from func.trackFeatures import threshold
from numpy.ma.core import cos, cosh, mean

'''
Created on Jan 27, 2015

@author: Tomasz
'''

class ChessboardDetector(object):
    '''
    classdocs
    '''
    
    '''
    promień skanowania wokół punktu w celu ustalenia czy jest wierzchołkiem szachowicy
    '''
    CornerScanRadius = 20
    
    '''
    Ilość punktów okręgu wokół punktu przy ustaleniu czy jest wierzchołkiem szachownicy
    np. 73 -> 360stopni/72 -> rozdielczość pół stopnia
    '''
    CornerScanResolution = 73
    
    
    #3D
    objectPoints = None
    
    #2D
    imagePoints = None
    
    def getBendMap(self,binary,board_w=10,board_h=7):
        '''
            znajdź maskę połaczenia szachownic
            binary - binarna mapa czarnych pól na zdjeciu uzyskana tresholdem
            board_w - szerokość pojedynczej szachownicy
            board_h - wysokośc pojedynczej szachownicy
        '''
        
        #wykrywanie zgiecia
        
        #erozja w celu rozdzielenia pól szachownicy
        size = 5
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        b3 = cv2.erode(binary,cross)
        
        #policzenie powierzchni etykiet
        labelMap = measure.label(b3,8,0)
        properties = measure.regionprops(labelMap)
        
        areas = [prop['convex_area'] for prop in properties]
        
        w = board_w+1
        h = board_h+1
        fields = h*w
        if h%2 == 0:
            bias = w%2
        else:
            bias = 0
        connected = w/2+bias
        fields -= connected
        
        print 'number of fields',fields
        print 'number of connected fields',connected
        print 'number of areas',len(areas)
        diff = len(areas)-fields
        
        areas.sort()
        areas.reverse()
        print 'areas',areas
        print 'diff',diff
        avg = sum(areas) / len(areas)
        print 'average', avg
        backgroundFields = []
        if diff>0:
            for d in range(diff):
                if areas[d]>avg:
                    backgroundFields.append(areas[d])
                    areas[d] = 0
        areas.sort()
        areas.reverse()
        obsoleteFields = []
        for i in range(connected):
            obsoleteFields.append(areas[i])
        
        edgeBinaryMap = np.zeros(labelMap.shape)
        
        for p in properties:
            area = p['convex_area']
            if area in obsoleteFields:
                edgeBinaryMap[labelMap==p['label']]=1
        
        return edgeBinaryMap
    
    def getChessCorners(self,binary,edgeBinaryMap,corners2Shifted):
        r = self.CornerScanRadius
        angles = np.linspace(0, 2*np.pi, self.CornerScanResolution)
        cosinuses = np.cos(angles) * r
        sinuses = np.sin(angles) *r
        
        counter = 0
        edgePoints = []
        chessCorners = []
        for p in corners2Shifted:
            values= []
            for c,s in zip(cosinuses,sinuses):
                x = int(p[0]+c)
                y = int(p[1]+s)
                if y >= binary.shape[1]:
                    values.append(0)
                    continue
                if x >= binary.shape[0]:
                    values.append(0)
                    continue
                try:
                    values.append(binary[x,y])
                except:
                    print 'exception'
                    
            edges = np.abs(np.diff(np.array(values,np.float16)))
            ec = int(np.sum(edges))
            
            # weryfikacja liczby krawędzi.
            # 4-> wierzchołek szachwnicy
            # 2-> inny wierzchołek
            row,col = draw.circle_perimeter(p[0],p[1],20)
            if row.max() > binary.shape[0]:
                continue
            if col.max() > binary.shape[1]:
                continue
            if ec ==2:
                if edgeBinaryMap[p[0],p[1]] == 1:
                    edgePoints.append(p)
            elif ec ==4:
                chessCorners.append(p)
                counter +=1
                
        shape = binary.shape
        p1,p2 = self.getBendLine(edgePoints, shape)
        leftMask = self.getLeftChessboardMask(shape, p1, p2)
                
        return chessCorners,leftMask
    
    def getBendLine(self,edgePoints,shape):
        '''
            zwraca dwa puntky wyznaczającą linie łaczenia szachownic
        '''
        line = measure.LineModel()
        line.estimate(np.array(edgePoints))
        xMax = shape[0]-1
        p1Y = int(line.predict_y(0))
        p2Y = int(line.predict_y(xMax))
        p1 = (p1Y,0)
        p2 = (p2Y,xMax)
        
        return p1,p2
    
    def getLeftChessboardMask(self,shape,p1,p2):
        '''
            p1,p2 - punkty wyznaczające linie łaczenia szachownic
        '''
        left = np.zeros(shape)
        left[p1[1],p1[0]] = 1
        left[p2[1],p2[0]] = 1
        left[0,0] = 1
        left[p2[1],0] = 1
        leftMask = morphology.convex_hull_image(left)
        
        return leftMask

    def getZoneCorners(self,corners,offset):
        '''
        corners - wierzchołki znalezione we współrzednych globalnego obrazu
        '''
        offsetX,offsetY = (offset[0],offset[1])
#         offsetX,offsetY,width,height = (1618, 63, 979, 910)
        corners2Shifted = []
        for c in corners:
            c = ( c[0] - offsetY, c[1]-offsetX )
            corners2Shifted.append(c)
            
        return corners2Shifted
    
    def splitCorners(self,chessCorners,leftMask):
        cornersLeft = []
        cornersRight = []
        maskLeft = np.zeros(leftMask.shape,np.uint8)
        maskRight = np.zeros(leftMask.shape,np.uint8)
        for p in chessCorners:
            if leftMask[p[0],p[1]]:
                cornersLeft.append(p)
                maskLeft[p[0],p[1]] = 1
            else:
                cornersRight.append(p)
                maskRight[p[0],p[1]] = 1
        return cornersLeft,cornersRight,maskLeft,maskRight
    
    def getWorldPoints(self,leftPoints,rightPoints):
        finalPointsL = np.zeros( (len(leftPoints),len(leftPoints[0]),3) )
        finalPointsR = np.zeros( (len(rightPoints),len(rightPoints[0]),3) )
        
        for row in range(len(leftPoints)):
            for col in range(len(leftPoints[row])):
                p =  (row,col,0)
                finalPointsL[row,col] = p
                
        for row in range(len(rightPoints)):    
            for col in range(len(rightPoints[row])):
                p =  (len(rightPoints),col,len(rightPoints)-(row))
                finalPointsR[row,col] = p
        
        finalPoints = np.append(finalPointsL,finalPointsR,0)
        final = np.append(leftPoints,rightPoints,0)

        return final, finalPoints
    
    def getImagePoints(self,points,shape,left=True):
        orderedPoints = []
        
        newPoints = list(points)
        while len(newPoints)>0:
            scanLinePoints = self.getScanLine(points,shape,left)
            p = self.orderPoints(scanLinePoints)
            orderedPoints.append( p )
            newPoints = self.removePointsFromList(points, scanLinePoints)
            
        return orderedPoints
    
    def orderPoints(self,points):
        orderedPoints = []
        py = [p[0] for p in points]
        m = max(py)
        while min(py)<=m:
            index = py.index(min(py))
            orderedPoints.append( points[ index ])
            py[index] = m+1 

        return orderedPoints
    
    def get_closest_to(self,origin,points):
        miN = 10000000
        start_point = None
        for p in points:
            cost = (p[0]-origin[0])**2 + (p[1]-origin[1])**2
            if cost < miN:
                miN = cost
                start_point = p
                
        return start_point
    
    def get_furthest_to(self,origin,points):
        maX = 0
        start_point = None
        for p in points:
            cost = (p[0]-origin[0])**2 + (p[1]-origin[1])**2
            if cost > maX:
                maX = cost
                start_point = p
                
        return start_point
    
    def getScanLine(self,points,shape,leftSide = True):
        if leftSide:
            start = self.get_closest_to((0,0),points)
            stop = self.get_furthest_to((0,0),points)
        else:
            start = self.get_closest_to((0,shape[1]),points)
            stop = self.get_furthest_to((0,shape[1]),points)

        
        extremePoints = [start,stop]
        line = measure.LineModel()
        line.estimate(np.array(extremePoints))
        re = line.residuals(np.array(points))
        re = re.tolist()
        i1 = re.index(max(re))
        i2 = re.index(min(re))
        p1 = points[i1]
        p2 = points[i2]
        if leftSide:
            end = self.get_closest_to((shape[0],0),[p1,p2])
        else:
            end = self.get_closest_to((shape[0],shape[1]),[p1,p2])
        
        scanPoints = [(start[0],start[1]),(end[0],end[1])]
        line2 = measure.LineModel()
        line2.estimate(np.array(scanPoints))
        re = line2.residuals(np.array(points))
        
        linePoints = []
        for i in range(len(points)):
            if abs(re[i]) < 5:
                linePoints.append(points[i]) 
        
        return linePoints
    
    def removePointsFromList(self,points,obsoletePoints):
        for o in obsoletePoints:
            points.remove(o)
        return points
    
    def find_objects(self,img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs = 150
        retval, binar = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
        size = 100
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        binar = cv2.dilate(binar,cross)
        binar = cv2.erode(binar,cross)
        
        lf = labeling.LabelFactory([])
        labelsMap = lf.getLabelsExternal(binar, 8, 0)
        return labelsMap
        
    def find_potential_corners(self,scene):
        
        gray = scene.gray
        
        dst = cv2.cornerHarris(gray,blockSize=5,ksize=3,k=0.04)
        harrisdst = dst.copy()
        
        size = 3
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        
        # zeby polaczyc tak na prawde te same wierzcholki
        # harris w tutorialach do cv3
        dst = cv2.dilate(dst,cross)
        ret, dst = cv2.threshold(dst,0.025*dst.max(),1,cv2.THRESH_BINARY)        
        
        bin = dst
        bb = np.zeros((gray.shape[0],gray.shape[1],3))
#         bb[dst==1]= (255,255,255)
#         bb =transform.rescale(bb, 0.33)
#         cv2.imshow('iii',bb)
#         cv2.waitKey()
#         cv2.destroyWindow('iii')
#         raise Exception()
        bb = scene.view.copy()
        
        lf = labeling.LabelFactory([])
        labelMap = lf.getLabelsExternal(dst, 8, 0)
#         bb[labelMap==0] = (0,255,0)
        corners = []
        
        #znajdz srodki z tych wierzcholkow z Harrisa
        for label in np.unique(labelMap):
            if label == -1:
                continue
            empty = np.zeros((gray.shape[0],gray.shape[1]))
            empty[labelMap == label] = 1
            M = cv2.moments(empty)
#             m = measure.moments(empty)
            
            cx = M['m10']/M['m00']
            cy = M['m01']/M['m00']
#             print cx,cy
            corners.append((int(cx),int(cy)))
        # znajdz główne obiekty na obrazie
        objectsLabelsMap = self.find_objects(scene.view)
        
        score = {}
        for (cx,cy) in corners:
            label = objectsLabelsMap[cy,cx]
            if label not in score.keys():
                score[label] = 0
            score[ label ] += 1
        max = 0    
        #znajdz etykiete na ktorej jest najwiecej wierzcholkow znalezioych
        for k,v in score.iteritems():
            if v>max:
                max = v
                maxI = k
        chessboardMap = np.where(objectsLabelsMap==maxI,1,0)
        
        # wyodrebnienie zony
        chess = np.nonzero(chessboardMap)
        chess = np.array([chess[0],chess[1]]).T
        
        BND2 = np.asarray([[(y,x) for (x,y) in chess]])
        x,y,w,h = cv2.boundingRect(BND2)
        print 'zone shape', x,y,w,h
        
        z = Zone(scene.view, x, y, w, h, chessboardMap)
        
        #narozniki - maska
        c1 = np.array(corners)
        map2 = np.zeros_like(chessboardMap)
#         map2[c1] = 1
        for c in corners:
            map2[c[1],c[0]] = 1
        
        #punkty wspolne maski i naroznikow
        cornersMap = cv2.bitwise_and(chessboardMap,map2)
        corners = np.nonzero(cornersMap)
        corners = np.array([corners[0],corners[1]]).T
        
        for (cx,cy) in corners:
            cv2.circle(bb,(int(cy),int(cx)),1,(255,255,0))
#             cv2.circle(img2,(int(cy),int(cx)),3,(0,255,255),1)
        
        
#         self.find_subcorners(harrisdst, corners, scene.view)
        
        return corners,z
     
    def __init__(self):
        '''
    
        '''
        pass
        

    def getCalibrationPoints(self,corners2Shifted,gray):
        
        ret, binary = cv2.threshold(gray,150,1,cv2.THRESH_BINARY_INV)
        shape = binary.shape
        
        board_w = 10
        board_h = 7
        edgeBinaryMap = self.getBendMap(binary,board_w,board_h)
        
        size = 15
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        edgeBinaryMap = cv2.dilate(edgeBinaryMap,cross)
        
        chessCorners, leftMask = self.getChessCorners(binary,edgeBinaryMap,corners2Shifted)

        print 'points:',len(chessCorners)
        print 'shoud be', board_w*board_h*2  
        
        cornersLeft,cornersRight,maskLeft,maskRight = self.splitCorners(chessCorners,leftMask)      
                
#         self.getDirections(leftMask,board_h)    
        
        leftPoints  = self.getImagePoints(cornersLeft, shape,True)
        rightPoints = self.getImagePoints(cornersRight,shape,False)
        finalPoints, finalWorldPoints = self.getWorldPoints(leftPoints,rightPoints)
        
        return finalPoints,finalWorldPoints
        
        
        
    def find_corners_opencv(self,leftMask,origin,edgeBinaryMap):
        '''
        not used -> znajduje wierzchołki przy użyciu opencv2 i odnalezieniu pojedynczych szachownic na obrazku
        '''
        left = origin.copy()
        
        left[edgeBinaryMap==1] = (255,255,255)
        left[leftMask==1] = (0,0,0)
        
        right = origin.copy()
        right[edgeBinaryMap==1] = (255,255,255)
        right[leftMask==0] = (0,0,0)
        
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        board_w = 20
        board_h = 14
        board_size = (10,7)
        found,corners = cv2.findChessboardCorners(gray,board_size,flags=cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FILTER_QUADS)
        
        if found:
            search_size = (11,11)
#             search_size = (11,11)
            zero_zone = (-1,-1)
            cv2.cornerSubPix(gray,corners,search_size,zero_zone,(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            cv2.drawChessboardCorners(gray, board_size, corners, True)
        else:
            print 'not found'
            
#         right = np.where(leftMask==0,origin,(0,0,0))
        f = self.writepath+ '%d_left_corners%d.jpg' % (7,2)
        cv2.imwrite(f,gray)
#         f = self.writepath+ '%d_right_corners%d.jpg' % (7,2)
#         cv2.imwrite(f,right)
        pass 
    
    def getDirections(self,leftMask,board_h):
        rho = 3
        theta = np.pi/360
        threshold = board_h-2
        
        lines = cv2.HoughLines(leftMask,rho,theta,threshold)
        lines2 = lines[0]
        lines2 = self.find_main_directions_lines(lines2)
        if lines2 is not None:
            thetas = []
            for (rho, theta) in lines2:
                thetas.append(theta)
            thetas.sort()
    
    def find_main_directions_lines(self,lines,linesExtreme='max'):
        '''
        not used - wyznacza linie hougha dla woerchołków szchownicy i znajduje głowne kierunki lini - wyznaczajce długość i szerokość
        '''
        thetas = []
        for (rho, theta) in lines:
            thetas.append(theta) 
        thetas = [t*180/np.pi for t in thetas] 
        thetas.sort()
        thetas.append(thetas[0]+360)
        
        #znajdz indeksy na liscie ktore sa brzegowe dla kieunkow
        t0 = np.array(thetas)
        d1 = np.diff(t0)
        edges1 = np.where(d1>5,1,0)
        indexes1 = np.nonzero(edges1)
        
        #policz ile jest lini w danym przedziale
        d2 = np.diff(np.array(indexes1))
        
        #znajdz dwa najmniej liczne kierunki (skosne)
        quantities = list(d2[0])
        quantities.insert(0, indexes1[0][0])
        
        quantities2 =list(quantities) 
        quantities2.sort()
        
        if linesExtreme == 'min':
            #wex minimum
            q1 = quantities2[0]
            q2 = quantities2[1]
                
            if q1 != q2:
                idx1 = quantities.index(q1)
                idx2 = quantities.index(q2)
            else:
                idx1 = quantities.index(q1)
                idx2 = quantities.index(q2,idx1+1)
            bias1 = 0
            bias2 = 0
            if idx1 == 0:
                bias1 = -1
            if idx2 == 0:
                bias2 = -1
                 
            i1 = indexes1[0][idx1]
            i2 = indexes1[0][idx2]
             
            range1 = thetas[i1-q1+1+bias1:i1+1]
            range2 = thetas[i2-q2+1+bias2:i2+1]
        
        elif linesExtreme:
        #wez max
            q1 = quantities2[-1]
            idx1 = quantities.index(q1)
            i1 = indexes1[0][idx1]
            bias1 = 0
            if idx1 == 0:
                bias1 = -1
            range1 = thetas[i1-q1+1+bias1:i1+1]
            range2 = []
        else:
            range1= thetas
            range2= []
        
        newLines = []
        for l in lines:
            r,theta = l
            theta = theta*180/np.pi
            
            if linesExtreme == 'max':
                if theta in range1:
                    newLines.append(l)
            else:
                if theta in range1:
                    continue
                if theta in range2:
                    continue
                newLines.append(l)
        return newLines
    
    def find_subcorners(self,dst,corners,img):
        '''
        not used - szuka wierzchołkow z wieksdza dokładnościa
        '''
        
        coords = feature.corner_peaks(dst, min_distance=10)
        print 'coords', coords
        coords_subpix = feature.corner_subpix(dst, coords, window_size=5)
        print 'subpix', coords_subpix
        img2 = img.copy()
        for coords in coords:
            cv2.circle(img2,(coords[1],coords[0]),3,(0,255,0),1)
        for coords in coords_subpix:
            if str(coords[1]) == str(float('nan')):
                print float('nan')
                continue
            cv2.circle(img2,(int(coords[1]),int(coords[0])),3,(255,0,0),1)
            
        
    