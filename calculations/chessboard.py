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
        
        bb = np.zeros((b3.shape[0], b3.shape[1], 3))
        bb[b3 == 1] = (255, 0, 255)
        cv2.imwrite('results/chessboardMap.jpg', bb)
        
        # policzenie powierzchni etykiet
        labelMap = measure.label(b3, 8, 0)
        properties = measure.regionprops(labelMap)
        
        areas = [prop['convex_area'] for prop in properties]
        
        
        w = board_w + 1
        h = board_h + 1
        fields = h * w
        if h % 2 == 0:
            bias = w % 2
        else:
            bias = 0
        connected = w / 2 + bias
        fields -= connected
        
        print 'number of fields', fields
        print 'number of connected fields', connected
        print 'number of areas', len(areas)
        diff = len(areas) - fields
        
        areas.sort()
        areas.reverse()
        print 'areas', areas
        print 'diff', diff
        avg = sum(areas) / len(areas)
        print 'average', avg
        backgroundFields = []
        if diff > 0:
            for d in range(diff):
                if areas[d] > avg:
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
        bb[edgeBinaryMap==1] = (0,0,10)
#         cv2.imwrite('results/binary_2.jpg',bb)
        
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
                    cv2.circle(bb,(p[1],p[0]),4,(255,0,255),-1 )
            elif ec == 4:
                cv2.circle(bb,(p[1],p[0]),4,(255,0,0),-1 )
                chessCorners.append(p)
                counter +=1
            else:
                cv2.circle(bb,(p[1],p[0]),4,(255,0,255),-1 )
                
        cv2.imwrite('results/corner_types_%d.jpg' % self.image_type,bb)
        shape = binary.shape
        print len(chessCorners)
        print len(edgePoints)
        
        if len(edgePoints) == 0:
            print 'empty!'
            
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
        self.bendLine = line
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
        
        print finalPointsL.shape
        print finalPointsR.shape
        
        for row in range(len(leftPoints)):
            for col in range(len(leftPoints[row])):
                p =  (row,col,0)
                finalPointsL[row,col] = p
                
        for row in range(len(rightPoints)):    
            for col in range(len(rightPoints[row])):
                p =  (len(rightPoints),col,len(rightPoints)-(row))
                finalPointsR[row,col] = p
        
        print finalPointsL.shape
        print finalPointsR.shape
        
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
        w = 10
        
        
        bend = self.bendLine
        re = bend.residuals(np.array(points))
        re = re.tolist()
        
        indexes = []
        for k in range(10):
            idx = re.index(min(re)) 
            indexes.append(idx)
            re[idx] = 10000
            
        linePoints = []
        for k in indexes:
            linePoints.append(points[k])
        
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
        size = 100
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        binar = cv2.dilate(binar,cross)
        binar = cv2.erode(binar,cross)
        
        bb = np.zeros_like(img)
        bb= np.where(binar == 1,255,0)
        
        cv2.imwrite('results/objects_binary.jpg',bb)        
        
        lf = labeling.LabelFactory([])
        labelsMap = lf.getLabelsExternal(binar, 8, 0)
        return labelsMap
        
    def find_potential_corners(self,scene):
        
        gray = scene.gray
        
        dst = cv2.cornerHarris(gray,blockSize=5,ksize=3,k=0.04)
        harrisdst = dst.copy()
        
        size = 5
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
        bb[dst>0] = (255,0,0)
        cv2.imwrite('results/Harris.jpg',bb)
        
        lf = labeling.LabelFactory([])
        labelMap = lf.getLabelsExternal(dst, 8, 0)
        bb[labelMap==0] = (0,255,0)
        corners = []
        
        print 'znajdź centroidy wierzchołków Harrisa'
        #znajdz srodki z tych wierzcholkow z Harrisa
#         inicesList = {}
#         shape = (gray.shape[0],gray.shape[1])
#         print len(np.unique(labelMap))
#         
#         for label in np.unique(labelMap):
#             inicesList[label] = []
#         del inicesList[-1]
#         for index, label in np.ndenumerate(labelMap):
#             if label == -1:
#                 continue
#             inicesList[label].append(index)
# 
#         gc.collect()
#                     
#         moments = cv2.moments
#         for label,value_list in inicesList.iteritems():
#             v = np.array(value_list).T
#             offset = (v[0].min(),v[1].min())
#              
#             shapeTMP = (v[0].max() - v[0].min() +1, v[1].max() - v[1].min() + 1) 
#              
#             empty = np.zeros(shapeTMP)
#             a = v[0]-offset[0]
#             b = v[1]-offset[1]
#              
#             c = np.array([a,b])
#             empty[(c[0],c[1])] = 1
#             M = moments(empty)
# #             m = measure.moments(empty)
#             cx = M['m10']/M['m00'] + offset[1]
#             cy = M['m01']/M['m00'] + offset[0]
#              
#             corners.append((int(cx),int(cy)))
                    
        print 'corners', corners
        corners = [(948, 5), (973, 6), (993, 11), (912, 20), (4332, 30), (1027, 64), (1009, 71), (994, 79), (952, 100), (930, 110), (955, 114), (2133, 129), (2207, 153), (998, 164), (2035, 178), (2286, 179), (2110, 201), (2360, 203), (2186, 226), (2438, 229), (2015, 248), (2262, 250), (2511, 253), (2088, 273), (2337, 275), (2589, 278), (2164, 297), (2412, 299), (2661, 302), (2239, 322), (1993, 322), (2487, 324), (2738, 328), (4504, 334), (2067, 344), (980, 343), (2313, 346), (991, 346), (2562, 348), (2810, 352), (1022, 356), (1034, 359), (2142, 368), (2388, 370), (2636, 373), (1974, 389), (2216, 392), (2462, 394), (2710, 397), (4455, 405), (2047, 413), (2290, 416), (2536, 418), (1584, 417), (2784, 421), (2120, 437), (2364, 440), (2609, 443), (2855, 446), (2194, 461), (1953, 460), (2437, 464), (2683, 467), (2026, 482), (2267, 485), (2511, 488), (2756, 490), (2924, 488), (2100, 506), (2341, 508), (2583, 511), (2828, 513), (1935, 525), (2172, 529), (2413, 532), (2656, 535), (2006, 549), (2245, 552), (2486, 555), (2729, 558), (2990, 566), (2079, 573), (2317, 575), (2558, 579), (2799, 583), (2151, 596), (1915, 595), (2390, 598), (2630, 602), (1986, 616), (2223, 619), (2461, 622), (2702, 625), (1822, 633), (2059, 638), (2295, 641), (2533, 645), (2772, 647), (2130, 661), (2366, 664), (2604, 668), (2202, 684), (2438, 687), (1887, 688), (2676, 691), (2273, 706), (2509, 710), (1957, 713), (2744, 715), (1875, 725), (2108, 724), (2343, 729), (2579, 733), (2029, 736), (1947, 748), (2183, 747), (2414, 751), (2650, 755), (1865, 756), (2101, 760), (2020, 772), (2249, 769), (2484, 774), (2719, 776), (1936, 783), (2172, 782), (3879, 798), (2092, 795), (1852, 795), (2555, 796), (2243, 804), (2009, 807), (2164, 817), (2624, 819), (2036, 818), (1925, 820), (2315, 827), (1841, 827), (2083, 831), (2237, 840), (2460, 837), (1999, 844), (2692, 842), (2386, 849), (2156, 854), (1913, 857), (2309, 863), (3807, 859), (2074, 868), (1827, 868), (2456, 871), (2230, 877), (1989, 882), (2380, 886), (2148, 892), (2526, 893), (1902, 895), (2302, 900), (1722, 901), (2064, 906), (1815, 905), (2452, 908), (2222, 914), (2597, 915), (1979, 921), (2375, 923), (2744, 925), (2140, 930), (2523, 930), (1890, 935), (2666, 934), (2296, 938), (2055, 945), (2448, 945), (1708, 947), (2595, 952), (2215, 953), (1967, 960), (2370, 961), (1706, 960), (2520, 968), (2131, 969), (2665, 974), (1879, 975), (2290, 977), (2443, 984), (2045, 985), (2592, 990), (2208, 993), (1955, 999), (2365, 1001), (2517, 1007), (2123, 1009), (2664, 1008), (2283, 1017), (2439, 1024), (2037, 1024), (2590, 1029), (2200, 1033), (2359, 1041), (2513, 1047), (2113, 1046), (2663, 1050), (2277, 1057), (2434, 1065), (2588, 1069), (2195, 1070), (2354, 1082), (2510, 1088), (2661, 1087), (2270, 1095), (2430, 1105), (2585, 1110), (2350, 1121), (2506, 1129), (2660, 1130), (2424, 1145), (2583, 1152), (2658, 1170), (2504, 1171), (2578, 1194), (2736, 1287), (870, 1410), (867, 1446), (3442, 1561), (3510, 1595), (3493, 1634), (2488, 1774), (3073, 1917), (3003, 2034), (2938, 2056), (3137, 2075), (2865, 2078), (3152, 2087), (2797, 2099), (3069, 2100), (2719, 2122), (2999, 2125), (2647, 2145), (2929, 2148), (3206, 2145), (3118, 2147), (2857, 2172), (2564, 2172), (3066, 2195), (2783, 2196), (2487, 2198), (3168, 2206), (3084, 2210), (2080, 2213), (2995, 2220), (2707, 2222), (2400, 2226), (2097, 2242), (2923, 2246), (2629, 2248), (2320, 2251), (2849, 2272), (2548, 2275), (3048, 2280), (2773, 2299), (2465, 2303), (2991, 2321), (2694, 2326), (2380, 2330), (2158, 2330), (2149, 2332), (2170, 2333), (3013, 2345), (2917, 2348), (2614, 2355), (2299, 2356), (2840, 2377), (2531, 2384), (2762, 2406), (2446, 2412), (2977, 2412), (2681, 2436), (2362, 2436), (2965, 2438), (2910, 2456), (2598, 2466), (2832, 2487), (2939, 2485), (2514, 2496), (2751, 2519), (2431, 2525), (2668, 2551), (3709, 2561), (2897, 2564), (2583, 2583), (2823, 2603), (2934, 2603), (2498, 2611), (2740, 2638), (2855, 2644), (3816, 2659), (2654, 2672), (2571, 2705), (2810, 2726), (3681, 2736), (3396, 2758), (2727, 2763), (3409, 2769), (3582, 2790), (2641, 2797), (2767, 2808), (3484, 2840), (3397, 2886), (2718, 2897)]

        print 'znajdź obiekty główne na obrazie'
        # znajdz główne obiekty na obrazie
        objectsLabelsMap = self.find_objects(scene.view)
        score = {}
        for [cx,cy] in corners:
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
        
        cv2.imwrite('results/zone.jpg',z.image)
        
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
            cv2.circle(bb,(int(cy),int(cx)),self.CornerScanRadius,(255,0,255))
#             cv2.circle(img2,(int(cy),int(cx)),3,(0,255,255),1)
        cv2.imwrite('results/corners_%d.jpg'% self.image_type, bb)
        
#         self.find_subcorners(harrisdst, corners, scene.view)
        self.origin = z.image
        return corners,z
     
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
        image_index = self.image_index 
        
        board_w = self.board_w
        board_h = self.board_h
        ret, binary = cv2.threshold(gray,150,1,cv2.THRESH_BINARY_INV)
        shape = binary.shape
        
        bb = self.origin.copy()
        bb[binary == 1] = (0,0,0)
        cv2.imwrite('results/mask_%d.jpg'% (image_index) ,bb)
        
        print 'board:', board_w, board_h
        board_w = 10
        board_h = 7
        edgeBinaryMap = self.getBendMap(binary,board_w,board_h)
        
        size = 15
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        edgeBinaryMap = cv2.dilate(edgeBinaryMap,cross)
        
        print 'before',len(corners2Shifted)
        
        chessCorners, leftMask = self.getChessCorners(binary,edgeBinaryMap,corners2Shifted)

        print 'points:',len(chessCorners)
        print 'should be', board_w*board_h*2  
        
        
        bb = self.origin.copy()
        
        
        cornersLeft,cornersRight,maskLeft,maskRight = self.splitCorners(chessCorners,leftMask) 
        bb[leftMask == 1] = (255,255,0)
        cv2.imwrite('results/mask1.jpg',bb)   
                
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
            
        
    