#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import time
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
import time
import gc

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
    
    origin = None
    image_type = 0
    
    leftTheta = None
    rightTheta = None
    bendLine = None
    image_index = 0
    filename = ''
    thresholdGetFields = 110 # binary_inv
    
    def getBendMap(self,binary,board_w=10,board_h=7):
        '''
            znajdź maskę połaczenia szachownic
            binary - binarna mapa czarnych pól na zdjeciu uzyskana tresholdem
            board_w - szerokość pojedynczej szachownicy
            board_h - wysokośc pojedynczej szachownicy
        '''
        
        #wykrywanie zgiecia
        
        #erozja w celu rozdzielenia pól szachownicy
        size = 7
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        b3 = cv2.erode(binary,cross)
        b3 = cv2.erode(b3,cross)
        b3 = cv2.erode(b3,cross)
        
        bb = np.zeros((b3.shape[0],b3.shape[1],3))
        bb[b3 == 1] = (255,0,255)
        cv2.imwrite('results/chessboardMap.jpg',bb)
        
        #policzenie powierzchni etykiet
        labelMap = measure.label(b3,8,0)
        properties = measure.regionprops(labelMap)
        
        areas = [prop['convex_area'] for prop in properties if prop['convex_area']>20]
        
        
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
#         print 'areas',areas
#         print 'diff',diff
        avg = sum(areas) / len(areas)
#         print 'average', avg
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
        bb = self.origin.copy()
        
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
            row,col = draw.circle_perimeter(p[0],p[1],self.CornerScanRadius)
            if row.max() > binary.shape[0]:
                continue
            if col.max() > binary.shape[1]:
                continue
            if ec == 2:
                cv2.circle(bb, ( p[1],p[0] ), 4, (255,255,0) ,-1 )
                if edgeBinaryMap[p[0],p[1]] == 1:
                    edgePoints.append(p)
                    cv2.circle(bb,(p[1],p[0]),4,(0,0,255),-1 )
            elif ec == 4:
                cv2.circle(bb,(p[1],p[0]),4,(255,0,0),-1 )
                chessCorners.append(p)
                counter +=1
            else:
                cv2.circle(bb,(p[1],p[0]),4,(255,0,255),-1 )
                
        cv2.imwrite('results/corner_types_%d_%s' % (self.image_index,self.filename),bb)
        shape = binary.shape
        print 'edge points', len(edgePoints)
        
        if len(edgePoints) == 0:
            print 'empty!'
            
        p1,p2 = self.getBendLine(edgePoints, shape)
        cv2.line(bb, (p1[0],p1[1]),(p2[0],p2[1]),(255,0,0),3)
        cv2.imwrite('results/q.jpg',bb)
        
        
        leftMask = self.getLeftChessboardMask(shape, p1, p2)
        
        return chessCorners,leftMask
    
    def getBendLine(self,edgePoints,shape):
        '''
            zwraca dwa puntky wyznaczającą linie łaczenia szachownic
        '''
        line = measure.LineModel()
        line.estimate(np.array(edgePoints))
        self.bendLine = line
        
        p1,p2 = self.getBendLineExtremePoints(line,shape)
        return p1,p2
    
    def getBendLineExtremePoints(self,line,shape):
        swap = False
        y1 = 0
        y2 = yMax = shape[0]-1
        xMax = shape[1]-1
        
        p1X = int(line.predict_y(y1))
        p1Y = y1
        if p1X < 0:
            p1X = 0
            p1Y = int(line.predict_x(p1X))
        elif p1X > xMax:
            swap = True
            p1X = xMax
            p1Y = int(line.predict_x(xMax))
            
        p2Y = y2
        p2X = int(line.predict_y(y2))
        if p2X > xMax:
            p2X = xMax
            p2Y = int(line.predict_x(xMax))
        elif p2X < 0:
            p2X = 0
            p2Y = int(line.predict_x(p2X))
        
        p1 = (p1X,p1Y)
        p2 = (p2X,p2Y)
        
        if swap:
            p1tmp = p2
            p2 = p1
            p1 = p1tmp
            
        return p1,p2
        
    
    def getLeftChessboardMask(self,shape,p1,p2):
        '''
            p1,p2 - punkty wyznaczające linie łaczenia szachownic
        '''
        left = np.zeros(shape)
        yMax,xMax = shape
        print 'shape', shape
        print 'p1:', p1
        print 'p2:', p2
        if p1[0] == 0:
            left[p1[1],p1[0]] = 1
            left[yMax-1,0] = 1
        else:
            left[p1[1],p1[0]] = 1
            left[0,0] = 1
            
        if p2[0] == xMax-1:
            left[p2[1],p2[0]] = 1
            left[yMax-1,xMax-1] = 1
        else:
            left[p2[1],p2[0]] = 1
            left[yMax-1,0] = 1

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
    
    def getGlobalCorners(self,corners,offset):
        '''
        corners - wierzchołki znalezione we współrzednych globalnego obrazu
        '''
        offsetX,offsetY = (offset[0],offset[1])
#         offsetX,offsetY,width,height = (1618, 63, 979, 910)
        corners2Shifted = []
        for row in corners:
            corners2Shifted.append([])
            for c in row:
                c = [ c[0] + offsetY, c[1]+offsetX ]
                corners2Shifted[-1].append(c)
            
        return corners2Shifted
    
    def splitCorners(self,chessCorners,leftMask):
        cornersLeft = []
        cornersRight = []
        maskLeft = np.zeros(leftMask.shape,np.uint8)
        maskRight = np.zeros(leftMask.shape,np.uint8)
        for p in chessCorners:
            if leftMask[p[0],p[1]] == 1:
                cornersLeft.append(p)
                maskLeft[p[0],p[1]] = 1
            else:
                cornersRight.append(p)
                maskRight[p[0],p[1]] = 1
        print 'left', len(cornersLeft)
        print 'right', len(cornersRight)
        return cornersLeft,cornersRight,maskLeft,maskRight
    
    def getWorldPoints(self,leftPoints,rightPoints):
        finalPointsL = np.zeros( (len(leftPoints) ,len(leftPoints[0]),3) )
        finalPointsR = np.zeros( (len(rightPoints),len(rightPoints[0]),3) )
        
        leftIMG = cv2.imread('results/pointss_left%s' % self.filename)
        rightIMG = cv2.imread('results/pointss_right%s' % self.filename)
        
        for row in range(len(leftPoints)):
            for col in range(len(leftPoints[row])):
#                 p =  (row,col,0)
                p =  (len(leftPoints)-(row)-1,col,0)
                
#                 #etykiety tekstowe
#                 ptxt = str( np.multiply(p, 20).tolist() )
#                 pt = (leftPoints[row][col][1],leftPoints[row][col][0])
#                 cv2.putText(leftIMG,ptxt,pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.75,(255,0,0) , 2)
                
                finalPointsL[row,col] = p
                
        for row in range(len(rightPoints)):    
            for col in range(len(rightPoints[row])):
#                 p =  (len(rightPoints),col,len(rightPoints)-(row))
                p =  (len(rightPoints),col,row+1)
                
#                 ptxt = str( np.multiply(p, 20).tolist() )
#                 pt = (rightPoints[row][col][1],rightPoints[row][col][0])
#                 cv2.putText(rightIMG,ptxt,pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.75,(255,0,0) , 2)
                
                finalPointsR[row,col] = p
                
        cv2.imwrite('results/pointss_left%s' % self.filename , leftIMG)
        cv2.imwrite('results/pointss_right%s' % self.filename, rightIMG)
        
        finalPoints = np.append(finalPointsL,finalPointsR,0)
        final = np.append(leftPoints,rightPoints,0)

        return final, finalPoints
    
    def getImagePoints(self,points,shape,left=True):
        orderedPoints = []
        from drawings.Draw import getColors
        
        newPoints = list(points)
        bb = np.zeros((shape[0],shape[1],3))
        bb = self.origin.copy()
        
        colors = getColors(140)
        i=0
        
        while len(newPoints)>0:
            
            scanLinePoints = self.getScanLine(points,shape,left)
            
            p = self.orderPoints(scanLinePoints)
            
#           punkty z etykietami  
#             for pp in p:
#                 pp = map(int,pp)
#                 cv2.circle(bb,(pp[1],pp[0]),8,colors[i],-1)
#                 cv2.putText(bb,str(i),(pp[1]-40,pp[0]), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5,(255,0,255),1 )
#                 i += 1
            
            orderedPoints.append( p )
            newPoints = self.removePointsFromList(points, scanLinePoints)
            
            
        if left:
            cv2.imwrite('results/pointss_left%s' % self.filename,bb)
        else:
            cv2.imwrite('results/pointss_right%s' % self.filename,bb)
        
        return orderedPoints
    
    def orderPoints(self,points):
        orderedPoints = []
        py = [p[1] for p in points]
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
        w = 10
        scan = np.zeros((shape[0],shape[1],3))
        scan = self.origin.copy()
        
        bend = self.bendLine
        p1,p2 = self.getBendLineExtremePoints(bend, shape)
        
        cv2.line(scan,(p1[0],p1[1]),(p2[0],p2[1]),(255,0,0),4 )
        
        
        re = bend.residuals(np.array(points))
        re = re.tolist()
        re = [abs(r) for r in re]
        
#         # etykiety odleglosci
#         for idx in range(len(re)):
#             ptxt = str(int(re[idx]))
#             pt = ( points[idx][1],points[idx][0] )
#             cv2.putText(scan,ptxt,pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1,(255,50,50) , 2)
        
        indexes = []
        for k in range(5):
            idx = re.index(min(re))
            indexes.append(idx)
            re[idx] = 100000
        
        markerPoints = []    
        for k in indexes:
            markerPoints.append(points[k])
            cv2.circle(scan,(points[k][1],points[k][0]),6,(0,255,0),4 )
            
        line = measure.LineModel()
        line.estimate(np.array(markerPoints))
        
        #policz odleglosci jeszcze raz
        re = line.residuals(np.array(points))
        re = re.tolist()
        re = [abs(r) for r in re]
        
        indexes = []
        for k in range(10):
            idx = re.index(min(re)) 
            indexes.append(idx)
            re[idx] = 100000
            
        linePoints = []
        for k in indexes:
            linePoints.append(points[k])
#             cv2.circle(scan,(points[k][1],points[k][0]),6,(0,255,0),4 )
            
#         window = cv2.namedWindow('a',cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('a',shape[0]/2,shape[1]/2)
#         cv2.imshow('a',scan)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
        
#         if leftSide:
#             start = self.get_closest_to((0,0),points)
#             stop = self.get_furthest_to((0,0),points)
#         else:
#             start = self.get_closest_to((0,shape[1]),points)
#             stop = self.get_furthest_to((0,shape[1]),points)
# 
#         
#         extremePoints = [start,stop]
#         line = measure.LineModel()
#         line.estimate(np.array(extremePoints))
#         re = line.residuals(np.array(points))
#         re = re.tolist()
#         i1 = re.index(max(re))
#         i2 = re.index(min(re))
#         p1 = points[i1]
#         p2 = points[i2]
#         if leftSide:
#             end = self.get_closest_to((shape[0],0),[p1,p2])
#         else:
#             end = self.get_closest_to((shape[0],shape[1]),[p1,p2])
#         
#         scanPoints = [(start[0],start[1]),(end[0],end[1])]
        
#         bb = self.origin.copy()
#         cv2.line(bb,(start[1],start[0]),(end[1],end[0]),(255,25,0),3 )
#         cv2.circle(bb,(start[1],start[0]),4,(0,255,0),3 )
#         f=  'results/line%d_%d_.jpg' % (points[0][0],points[0][1])
#         cv2.imwrite(f,bb)

        
#         line2 = measure.LineModel()
#         line2.estimate(np.array(scanPoints))
#         theta = line2.params[1]
#         print theta
#         re = line2.residuals(np.array(points))
#         
#         linePoints = []
#         for i in range(len(points)):
#             if abs(re[i]) < 5:
#                 linePoints.append(points[i]) 
        
        return linePoints
    
    def removePointsFromList(self,points,obsoletePoints):
        for o in obsoletePoints:
            points.remove(o)
        return points
    
    def find_objects(self,img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs = 180
        thrs = self.chesboardTreshold
        retval, binar = cv2.threshold(gray, thrs, 1, cv2.THRESH_BINARY)
        size = 101
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        binar = cv2.dilate(binar,cross)
        binar = cv2.erode(binar,cross)
        
        bb = np.zeros_like(img)
        bb= np.where(binar == 1,255,0)
        
        print 'results/objects_binary_%s'% self.filename
        cv2.imwrite('results/objects_binary_%s'% self.filename,bb)        
        
        lf = labeling.LabelFactory([])
        labelsMap = lf.getLabelsExternal(binar, 8, 0)
        return labelsMap
    
    def getCentroidHarris(self,binary):
        dst = binary
        
        print 'etykietuj wierzcolki'
        lf = labeling.LabelFactory([])
        labelMap = lf.getLabelsExternal(dst, 8, 0)
        corners = []
        
        print 'znajdź środki ciężkości wierzchołków Harrisa'
        #znajdz srodki z tych wierzcholkow z Harrisa
        inicesList = {}
        shape = (dst.shape[0],dst.shape[1])
         
        for label in np.unique(labelMap):
            inicesList[label] = []
        del inicesList[-1]
        for index, label in np.ndenumerate(labelMap):
            if label == -1:
                continue
            inicesList[label].append(index)
 
        gc.collect()
                     
        moments = cv2.moments
        for label,value_list in inicesList.iteritems():
            v = np.array(value_list).T
            offset = (v[0].min(),v[1].min())
              
            shapeTMP = (v[0].max() - v[0].min() +1, v[1].max() - v[1].min() + 1) 
              
            empty = np.zeros(shapeTMP)
            a = v[0]-offset[0]
            b = v[1]-offset[1]
              
            c = np.array([a,b])
            empty[(c[0],c[1])] = 1
            M = moments(empty)
#             m = measure.moments(empty)
            cx = M['m10']/M['m00'] + offset[1]
            cy = M['m01']/M['m00'] + offset[0]
              
            corners.append((int(cy),int(cx)))
        
        return corners
    
    def getHitScore(self,points,labelsMap):
        score = {}

        for [cy,cx] in points:
            label = labelsMap[(cy,cx)]
            if label not in score.keys():
                score[label] = 0
            score[ label ] += 1
            
#         print 'sceore', score
        max = 0    
        #znajdz etykiete na ktorej jest najwiecej wierzcholkow znalezioych
        for k,v in score.iteritems():
            if k == -1:
                continue
            if v>max:
                max = v
                maxI = k
        return maxI
    
    def getChessboardZone(self,scene, corners,objectsLabelsMap):
        maxI = self.getHitScore(corners,objectsLabelsMap)
        chessboardMap = np.where(objectsLabelsMap==maxI,1,0)
        # wyodrebnienie zony
        chess = np.nonzero(chessboardMap)
        chess = np.array([chess[0],chess[1]]).T
        
        BND2 = np.asarray([[(y,x) for (x,y) in chess]])
        x,y,w,h = cv2.boundingRect(BND2)
        chessboardMap_weaker = np.zeros_like(chessboardMap)
        chessboardMap_weaker[y:y+h,x:x+w] = 1 
        print 'chessboard zone shape', x,y,w,h
        
        z = Zone(scene.view, x, y, w, h, mask=None)
        
        return z,chessboardMap_weaker
    
    def filterCorners(self,corners,mask):
        #narozniki - maska
            
        map2 = np.zeros_like(mask)
#         map2[c1] = 1
        for c in corners:
            map2[(c[0],c[1])] = 1
        
        #punkty wspolne maski i naroznikow
        cornersMap = cv2.bitwise_and(mask,map2)
        corners = np.nonzero(cornersMap)
        corners = np.array([corners[0],corners[1]]).T
        
        return corners
        
    def find_potential_corners(self,scene):
        
        gray = scene.gray
        
        dst = cv2.cornerHarris(gray,blockSize=5,ksize=3,k=0.04)
        harrisdst = dst.copy()
        
        print 'znajdź obiekty główne na obrazie'
        # znajdz główne obiekty na obrazie
        objectsLabelsMap = self.find_objects(scene.view)
        
        size = 7
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        
        # zeby polaczyc tak na prawde te same wierzcholki
        # harris w tutorialach do cv3
        dst = cv2.dilate(dst,cross)
        ret, dst = cv2.threshold(dst,0.025*dst.max(),1,cv2.THRESH_BINARY)
#         ret, dst = cv2.threshold(dst,0.025*dst.max(),1,cv2.THRESH_TOZERO)  
#         for index, value in np.ndenumerate(dst):
#             if value == -1:
#                 pass
        indieces =  np.nonzero(dst)
        corners = np.array([indieces[0],indieces[1]]).T
        
        z, chessboardMap = self.getChessboardZone(scene, corners, objectsLabelsMap)
        
        corners = self.filterCorners(corners, chessboardMap)
        dst2 = np.zeros_like(chessboardMap)
        
        ct = corners.T
        dst2[(ct[0],ct[1])] = 1
        
        bb = np.zeros((gray.shape[0],gray.shape[1],3))
        bb = scene.view.copy()
        bb[chessboardMap>0] = (0,255,0)
        bb[dst2>0] = (255,0,0)
        
        print 'results/Harris_%s' % self.filename
        cv2.imwrite('results/Harris_%s' % self.filename, bb)
        
#         corners = [(960, 2065), (956, 2276), (965, 2607), (972, 2225), (977, 2011), (987, 2173), (999, 2546), (1005, 2120), (1014, 2276), (1014, 2208), (1022, 2067), (1033, 2235), (1038, 2666), (1040, 2012), (1042, 2199), (1049, 2177), (1054, 2263), (1064, 2197), (1067, 2124), (1085, 2069), (1096, 2243), (1104, 2014), (1112, 2181), (1131, 2128), (1150, 2072), (1153, 2231), (1161, 2164), (1164, 2227), (1170, 2015), (1179, 2706), (1180, 2189), (1189, 2151), (1189, 2162), (1198, 2131), (1212, 2148), (1218, 2074), (1220, 2804), (1225, 2572), (1238, 2020), (1248, 2196), (1267, 2130), (1265, 2670), (1263, 2907), (1286, 2077), (1302, 2179), (1306, 2768), (1304, 3004), (1309, 2540), (1342, 2145), (1348, 2867), (1348, 3108), (1350, 2635), (1375, 2091), (1390, 2967), (1389, 3206), (1391, 2732), (1395, 2506), (1417, 2652), (1421, 2140), (1424, 2383), (1433, 2830), (1431, 3066), (1433, 2601), (1433, 3310), (1474, 2697), (1474, 3166), (1474, 2928), (1474, 3409), (1474, 2476), (1484, 2231), (1508, 2353), (1514, 2568), (1515, 3026), (1515, 2793), (1516, 3266), (1518, 3514), (1555, 2663), (1556, 2890), (1556, 2444), (1557, 3124), (1558, 3366), (1555, 3883), (1560, 3613), (1594, 2536), (1595, 2758), (1597, 2987), (1599, 3223), (1601, 3467), (1616, 2312), (1631, 2415), (1634, 2630), (1636, 2853), (1638, 3083), (1641, 3322), (1643, 3567), (1671, 2505), (1674, 2723), (1676, 2949), (1680, 3181), (1683, 3421), (1687, 3666), (1709, 2384), (1710, 2597), (1714, 2817), (1717, 3044), (1721, 3279), (1725, 3520), (1736, 2260), (1746, 2474), (1750, 2690), (1754, 2911), (1758, 3140), (1763, 3376), (1765, 3620), (1785, 2565), (1790, 2782), (1795, 3006), (1799, 3237), (1805, 3475), (1825, 2657), (1830, 2875), (1831, 2332), (1835, 3100), (1840, 3333), (1847, 3571), (1864, 2748), (1870, 2969), (1873, 2421), (1876, 3196), (1882, 3430), (1887, 2303), (1904, 2840), (1910, 3062), (1913, 2512), (1917, 3291), (1922, 3527), (1928, 2395), (1943, 2932), (1941, 2277), (1950, 3156), (1953, 2604), (1957, 3387), (1969, 2488), (1975, 2804), (1983, 3024), (1986, 2369), (1991, 3250), (1993, 2696), (1999, 3481), (2002, 2247), (2010, 2581), (2016, 2899), (2023, 3117), (2028, 2463), (2033, 2788), (2031, 3344), (2046, 2341), (2051, 2675), (2053, 2986), (2060, 2218), (2063, 3209), (2070, 2558), (2073, 2880), (2071, 3438), (2089, 2437), (2092, 2768), (2105, 3303), (2108, 2313), (2112, 2653), (2113, 2973), (2127, 2185), (2132, 2534), (2133, 2862), (2147, 3394), (2152, 2410), (2153, 2748), (2153, 3065), (2172, 2283), (2174, 2630), (2174, 2956), (2192, 3159), (2190, 2154), (2196, 2508), (2195, 2844), (2214, 3051), (2218, 2382), (2217, 2727), (2232, 3252), (2237, 2940), (2240, 2252), (2240, 2607), (2255, 3146), (2253, 3454), (2260, 2825), (2262, 2482), (2269, 3345), (2278, 3036), (2283, 2705), (2286, 2353), (2287, 1976), (2295, 3241), (2302, 2922), (2306, 2582), (2309, 2222), (2320, 3132), (2326, 2805), (2331, 2455), (2335, 3336), (2345, 3020), (2351, 2683), (2354, 2323), (2361, 3230), (2370, 2904), (2376, 2558), (2387, 3119), (2395, 2784), (2397, 3327), (2401, 2430), (2414, 3004), (2420, 2660), (2429, 3217), (2440, 2885), (2444, 2531), (2456, 3104), (2466, 2763), (2469, 3316), (2485, 2987), (2491, 2640), (2499, 3205), (2512, 2866), (2529, 3088), (2535, 2742), (2536, 3306), (2558, 2970), (2572, 3192), (2584, 2850), (2604, 3073), (2613, 3295), (2631, 2951), (2648, 3178), (2680, 3060), (2687, 3284), (2724, 3163), (2782, 3389), (2896, 3371)]
#         corners = [(1089, 2878), (1122, 1934), (1165, 1759), (1202, 1722), (1230, 2696), (1231, 1823), (1250, 2630), (1255, 1707), (1264, 1484), (1274, 2553), (1297, 2727), (1296, 2482), (1299, 1577), (1316, 2662), (1315, 1456), (1322, 2400), (1339, 2592), (1338, 1676), (1345, 2325), (1363, 2519), (1372, 1868), (1373, 2236), (1379, 2692), (1387, 2445), (1398, 2156), (1403, 2625), (1412, 2368), (1427, 2556), (1427, 2062), (1438, 2288), (1452, 2484), (1454, 1975), (1466, 2206), (1466, 2654), (1477, 2410), (1487, 2590), (1493, 2120), (1504, 2333), (1512, 2521), (1522, 2032), (1531, 2254), (1538, 2449), (1543, 1645), (1544, 2621), (1553, 1941), (1559, 2172), (1564, 2375), (1569, 2555), (1588, 2088), (1585, 1848), (1592, 2299), (1589, 3425), (1596, 2486), (1613, 2654), (1618, 2000), (1614, 2686), (1620, 2221), (1622, 2415), (1627, 2586), (1650, 2522), (1650, 1909), (1650, 2139), (1650, 2342), (1677, 2453), (1678, 2267), (1680, 2055), (1679, 1818), (1704, 2382), (1701, 2555), (1707, 2189), (1711, 1969), (1728, 2489), (1733, 2310), (1738, 2108), (1744, 1879), (1747, 1547), (1756, 2421), (1762, 2234), (1769, 2025), (1784, 2351), (1781, 2522), (1779, 1787), (1787, 1578), (1793, 2157), (1796, 1586), (1801, 1938), (1805, 2458), (1814, 2278), (1813, 2567), (1824, 2077), (1822, 2556), (1824, 1613), (1834, 2390), (1835, 1849), (1844, 2203), (1851, 2490), (1856, 1995), (1862, 2320), (1867, 1760), (1875, 2127), (1879, 2426), (1890, 1909), (1892, 2247), (1907, 2047), (1908, 2358), (1910, 2555), (1924, 1820), (1923, 2173), (1940, 1965), (1937, 2290), (1941, 2493), (1956, 2097), (1962, 1731), (1969, 2218), (1971, 2426), (1975, 1881), (1973, 2629), (1988, 2018), (2001, 2358), (2000, 2145), (2004, 2564), (2010, 1793), (2023, 1937), (2033, 2287), (2033, 2698), (2033, 2068), (2036, 2497), (2044, 1705), (2058, 1853), (2066, 2213), (2067, 2429), (2068, 2636), (2067, 1991), (2094, 1766), (2101, 2358), (2099, 2775), (2101, 2138), (2101, 2569), (2102, 1909), (2135, 2501), (2134, 2709), (2133, 1678), (2135, 2285), (2136, 2059), (2135, 1481), (2138, 1827), (2146, 1457), (2161, 2847), (2169, 2643), (2169, 1472), (2171, 2210), (2169, 2431), (2173, 1979), (2175, 1738), (2201, 2784), (2205, 2359), (2204, 2575), (2208, 2132), (2210, 1896), (2212, 1652), (2230, 2925), (2238, 2718), (2240, 2506), (2242, 2284), (2246, 2052), (2250, 1809), (2258, 1546), (2269, 2861), (2275, 2651), (2277, 2434), (2281, 2206), (2285, 1969), (2287, 1721), (2296, 3119), (2297, 3000), (2308, 2795), (2312, 2582), (2315, 2360), (2320, 2127), (2326, 1883), (2339, 2939), (2346, 2728), (2350, 2510), (2355, 2283), (2362, 2044), (2367, 1798), (2379, 2874), (2385, 2659), (2390, 2436), (2397, 2203), (2398, 3227), (2404, 1958), (2410, 3015), (2418, 2807), (2425, 2588), (2432, 2360), (2440, 2120), (2443, 1872), (2449, 2955), (2459, 2739), (2467, 2515), (2475, 2281), (2483, 2035), (2491, 2886), (2501, 2669), (2510, 2439), (2520, 2199), (2527, 1953), (2531, 2822), (2544, 2596), (2555, 2360), (2565, 2114), (2575, 2749), (2590, 2520), (2602, 2278), (2607, 2029), (2619, 2680), (2636, 2442), (2648, 2195), (2669, 2601), (2685, 2361), (2695, 2113), (2716, 2527), (2734, 2278), (2770, 2442), (2779, 2194), (2819, 2363), (3031, 2281)]
        if self.filename == '0_1.jpg':
            corners = [(64, 2354), (85, 1804), (183, 2347), (196, 2297), (214, 2362), (227, 2313), (239, 2266), (252, 2380), (261, 2329), (273, 2280), (284, 2395), (288, 2231), (296, 2345), (301, 1619), (307, 2295), (318, 2246), (324, 2414), (332, 2362), (329, 2201), (343, 2311), (353, 2261), (364, 2214), (358, 2431), (369, 2379), (376, 1741), (377, 2168), (379, 2327), (390, 2277), (395, 1699), (394, 1788), (398, 2229), (400, 2451), (407, 2396), (408, 2182), (412, 1749), (416, 2344), (415, 1842), (417, 2139), (425, 2293), (430, 1800), (430, 1709), (435, 2245), (432, 1887), (436, 2470), (444, 2197), (445, 2415), (450, 1760), (452, 2152), (450, 1851), (454, 2361), (452, 1940), (463, 2310), (464, 2107), (468, 1900), (469, 1812), (466, 2060), (472, 2260), (469, 1984), (472, 1722), (480, 2212), (479, 2491), (484, 2433), (486, 1863), (486, 1949), (488, 2166), (488, 1773), (489, 2034), (492, 2379), (496, 2121), (501, 2327), (504, 1996), (507, 1825), (505, 1913), (503, 2073), (509, 2276), (505, 1734), (516, 2228), (516, 2511), (522, 1963), (524, 2181), (522, 2044), (524, 2452), (525, 1785), (525, 1876), (531, 2135), (532, 2396), (537, 2088), (539, 2344), (543, 1926), (541, 2011), (545, 1838), (547, 2293), (549, 1747), (548, 2593), (554, 2243), (558, 2058), (561, 2195), (561, 1976), (564, 1889), (565, 2473), (566, 1798), (568, 2149), (572, 2415), (578, 2024), (575, 2102), (579, 2361), (582, 1940), (584, 1851), (585, 2310), (585, 1758), (592, 2260), (596, 2072), (598, 2212), (599, 1989), (604, 2164), (602, 1902), (606, 1811), (607, 2491), (609, 2117), (613, 2435), (616, 2039), (620, 1953), (619, 2380), (626, 2327), (624, 1864), (631, 2276), (633, 2086), (630, 1773), (637, 2227), (637, 2004), (642, 1916), (643, 2179), (646, 1824), (648, 2131), (654, 2053), (652, 2454), (661, 2398), (660, 1968), (665, 1878), (666, 2345), (672, 2293), (668, 1785), (671, 2101), (677, 2243), (677, 2018), (681, 2195), (682, 1930), (688, 1838), (685, 2147), (693, 2068), (699, 1982), (702, 2416), (706, 1892), (707, 2363), (709, 2116), (712, 2311), (717, 2260), (715, 1800), (716, 2033), (721, 2211), (724, 1945), (725, 2162), (730, 1852), (732, 2082), (741, 1997), (748, 1906), (746, 2382), (749, 2132), (754, 2330), (756, 2278), (752, 1813), (758, 2047), (760, 2228), (762, 2178), (765, 1959), (772, 1867), (774, 2098), (783, 2011), (790, 2146), (790, 1921), (799, 2296), (796, 2347), (799, 2063), (801, 2244), (807, 1975), (804, 2194), (815, 2113), (814, 1883), (824, 2027), (826, 1778), (830, 2163), (830, 1934), (838, 2314), (840, 2079), (842, 2262), (843, 2211), (849, 1991), (856, 2129), (864, 2040), (872, 2179), (881, 2096), (884, 2278), (886, 2226), (896, 2143), (913, 2197), (927, 2243)]
        elif self.filename == '0_2.jpg':
            corners = [(1426, 3173), (1436, 2602), (1443, 2656), (1600, 3113), (1600, 3209), (1606, 3011), (1655, 3157), (1654, 3065), (1656, 3253), (1712, 3107), (1711, 3016), (1714, 3202), (1720, 3302), (1763, 2972), (1767, 3058), (1771, 3150), (1775, 3248), (1779, 3348), (1821, 3011), (1819, 2926), (1827, 3101), (1832, 3196), (1838, 3295), (1846, 3400), (1847, 3751), (1869, 2884), (1875, 2966), (1881, 3053), (1888, 3145), (1894, 3242), (1898, 2243), (1901, 3344), (1908, 3449), (1914, 2755), (1922, 2838), (1927, 2921), (1934, 3006), (1934, 3693), (1942, 3096), (1950, 3190), (1958, 3290), (1967, 3395), (1964, 2408), (1966, 2464), (1968, 2526), (1969, 2580), (1970, 2639), (1971, 2689), (1972, 2745), (1971, 2794), (1978, 2876), (1978, 3504), (1986, 2960), (1995, 3047), (2004, 3140), (2006, 3652), (2014, 3237), (2023, 3339), (2027, 2441), (2028, 2501), (2028, 2383), (2029, 2559), (2029, 2616), (2028, 2830), (2030, 2672), (2030, 2726), (2030, 2779), (2034, 3446), (2037, 2914), (2047, 3000), (2044, 3557), (2050, 3827), (2057, 3090), (2068, 3185), (2079, 3284), (2090, 3389), (2090, 2818), (2088, 2868), (2091, 2764), (2093, 2709), (2094, 2477), (2093, 2597), (2093, 2653), (2093, 2416), (2094, 2537), (2098, 2954), (2102, 3499), (2109, 3042), (2118, 3616), (2121, 3134), (2133, 3231), (2146, 3333), (2148, 2907), (2152, 2856), (2154, 2803), (2156, 2748), (2157, 2692), (2159, 2635), (2161, 2994), (2159, 3441), (2160, 2575), (2161, 2515), (2164, 2456), (2173, 3085), (2173, 3554), (2186, 3179), (2187, 3671), (2200, 3279), (2215, 3384), (2213, 2946), (2215, 2896), (2217, 2843), (2220, 2788), (2222, 2732), (2224, 3036), (2225, 2674), (2230, 3494), (2228, 2614), (2230, 2554), (2230, 2492), (2238, 3129), (2245, 3611), (2253, 3226), (2269, 3328), (2274, 2987), (2276, 3870), (2279, 2937), (2283, 2884), (2285, 3435), (2286, 2829), (2289, 3079), (2290, 2772), (2294, 2714), (2293, 4051), (2297, 2654), (2301, 2593), (2302, 3549), (2305, 3174), (2306, 2533), (2307, 4027), (2322, 3274), (2319, 3666), (2339, 3378), (2342, 3028), (2346, 2978), (2350, 2926), (2354, 2871), (2357, 3489), (2356, 3122), (2359, 2814), (2364, 2756), (2369, 2696), (2373, 2634), (2374, 3220), (2373, 3606), (2376, 2572), (2392, 3322), (2406, 3071), (2411, 3430), (2413, 3021), (2418, 2968), (2424, 3168), (2424, 2913), (2430, 2856), (2431, 3541), (2436, 2798), (2441, 2739), (2444, 3268), (2446, 2677), (2454, 2617), (2464, 3373), (2477, 3115), (2482, 3065), (2483, 3483), (2489, 3011), (2494, 3215), (2496, 2956), (2492, 3627), (2502, 2900), (2509, 2842), (2516, 2782), (2516, 3317), (2523, 2720), (2528, 2657), (2538, 3422), (2545, 3160), (2553, 3110), (2560, 3057), (2566, 3263), (2567, 3003), (2576, 2945), (2584, 2887), (2587, 3368), (2592, 2826), (2601, 2764), (2613, 2703), (2618, 3205), (2626, 3156), (2634, 3103), (2639, 3309), (2643, 3048), (2651, 2992), (2661, 2933), (2670, 2873), (2680, 2811), (2688, 2748), (2689, 3252), (2697, 3205), (2706, 3149), (2715, 3098), (2727, 3037), (2736, 2983), (2747, 2919), (2758, 2861), (2767, 3300)]
        else:
            corners = self.getCentroidHarris(dst2)
        print 'corners', corners
        
#         self.find_subcorners(harrisdst, corners, scene.view)
        self.origin = z.image
        self.scene = scene.view
        return corners, z
     
    def __init__(self,chesboardDetectionTreshold=180, (w,h) = (10,7), diameter=20):
        '''
    
        '''
        self.chesboardTreshold = chesboardDetectionTreshold
        self.CornerScanRadius = 20
        self.board_w = 10
        self.board_h = 7
        pass
        

    def getPoints(self,corners2Shifted,gray):
        self.image_index +=1
        
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        board_w = self.board_w
        board_h = self.board_h
        
        gamma_correction = 1.55
        img_tmp = gray / 255.0
        cv2.pow(img_tmp, gamma_correction, img_tmp)
        gamma = img_tmp * 255.0
        gamma = gamma.astype('uint8')
        
        
        cv2.imwrite('results/gray.jpg',gray)
        cv2.imwrite('results/gamma.jpg',gamma)
        gray = gamma
        ret, binary = cv2.threshold(gray,self.thresholdGetFields,1,cv2.THRESH_BINARY_INV)
        size = 5
        #filtrowanie zanieczyszczen
        cross = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(size,size))
        binary = cv2.erode(binary,cross)
        binary = cv2.dilate(binary,cross).astype('uint8')
        
        binary = cv2.dilate(binary,cross).astype('uint8')
        binary = cv2.erode(binary,cross)
        
        #filtruj znowu
        binaryF = cv2.dilate(binary,cross)
        binaryF = cv2.dilate(binaryF,cross)
        
        lf = labeling.LabelFactory([])
        labelsMap = lf.getLabelsExternal(binaryF, 8, 0)
        
        maxLabel = self.getHitScore(corners2Shifted, labelsMap)
        finalChessboarMap = np.where(labelsMap == maxLabel,1,0).astype('uint8')
        
        binary = cv2.bitwise_and(binary,finalChessboarMap)
        
        shape = binary.shape
        
        bb = self.origin.copy()
        bb[binaryF == 1] = (0,0,0)
        for c in corners2Shifted:
            cv2.circle(bb,(c[1],c[0]),5,(255,0,0),-1)
            
        print 'results/mask_%s'% (self.filename)
        cv2.imwrite('results/mask_%s'% (self.filename) ,bb)
        
        print 'board:', board_w, board_h
        board_w = 10
        board_h = 7
        edgeBinaryMap = self.getBendMap(binary,board_w,board_h)
        
        size = 21
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        edgeBinaryMap = cv2.dilate(edgeBinaryMap,cross)
        
        bb = self.origin.copy()
        
        bb[binary==1] = (50,50,200)
        bb[edgeBinaryMap==1] = (255,10,255)
        for p in corners2Shifted:
            cv2.circle(bb,(p[1],p[0]),1,(255,0,0),-1 )
            pass
        
        print 'results/processed_%s' % (self.filename)
        cv2.imwrite('results/processed_%s' % (self.filename),bb)
        
        chessCorners, leftMask = self.getChessCorners(binary,edgeBinaryMap,corners2Shifted)

        print 'points:',len(chessCorners)
        print 'should be', board_w*board_h*2  
        
        
        bb = self.origin.copy()
        
        
        cornersLeft,cornersRight,maskLeft,maskRight = self.splitCorners(chessCorners,leftMask) 
        bb[leftMask == 1] = (255,255,0)
        for coords in cornersRight:
            cv2.circle(bb,(coords[1],coords[0]),3,(0,255,0),3)
        cv2.imwrite('results/mask_%s' % (self.filename),bb)   
                
#         self.getDirections(leftMask,board_h)    
        
        leftPoints  = self.getImagePoints(cornersLeft, shape,True)
        
        l = np.array(leftPoints)
        
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
            
        
    