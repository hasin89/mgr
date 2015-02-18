# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
from calculations import triangulation,DecompPMat, labeling
from scene.edge import edgeMap
import pickle
import cv2
import func.markElements as mark
from scene.mirrorDetector import mirrorDetector
from scene.zone import Zone
from scene.objectDetector2 import objectDetector2
from calculations.calibration import CalibrationFactory
from scene.scene import Scene
from scene import edgeDetector, zone
from scene.qubic import QubicObject
import sys,os
from skimage import measure
from skimage import feature
from skimage import morphology,draw,transform

from drawings.Draw import getColors
import skimage
from func.trackFeatures import threshold
from numpy.ma.core import cos, cosh, mean
from math import sqrt
from numpy.core.numeric import NaN 

class edgeDetectionTest(unittest.TestCase):

    writepath = ''
    
    def setUp(self):
        np.set_printoptions(precision=4)

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
       
    def calibrate(self,numBoards):
        numBoards = 2#14
        board_w = 20
        board_h = 14
        flag = False
        
        CF = CalibrationFactory(numBoards,board_w,board_h,flag,self.writepath+'/calibration/src/'+str(self.i)+'/',self.writepath+'calibration/' )
        CF.showCamerasPositions()
        
        mtx, dist, rvecs, tvecs = CF.mtx,CF.dist,CF.rvecs,CF.tvecs
        print 'cam'
        print mtx
        print rvecs
        print tvecs
            
        error = CF.reprojectPoints(CF.filenames)
        
#         print 'errors',error
#         print 'image points',CF.imagePoints
        CF.objectPoints
        
        fundamental = cv2.findFundamentalMat(CF.imagePoints[0],CF.imagePoints[1],cv2.FM_8POINT)
        
        retval, H1,H2 = cv2.stereoRectifyUncalibrated(CF.imagePoints[0],CF.imagePoints[1],fundamental[0],CF.shape)
        print 'fundamental', fundamental[0]
        print retval
        print H1
        print H2
        
        fundamental = fundamental[0]
        h, w = CF.shape
#         overlay1 = cv2.warpPerspective(CF.img1, H1, (w, h))
#         f = self.writepath+ 'calibration/1_recified_%d.jpg' % (self.i)
#         print 'savaing to ' + f
#         cv2.imwrite(f, overlay1)
#         overlay = cv2.warpPerspective(CF.img2, H2, (w, h))
#         f = self.writepath+ 'calibration/2_recified_%d.jpg' % (self.i)
#         print 'savaing to ' + f
#         cv2.imwrite(f, overlay)
        
        lines1 = cv2.computeCorrespondEpilines(CF.imagePoints[0],1,fundamental)
        lines1 = lines1.reshape(-1,3)
        
        lines2 = cv2.computeCorrespondEpilines(CF.imagePoints[1],2,fundamental)
        lines2 = lines2.reshape(-1,3)
        
        img2 = CF.img2.copy()
        colors = getColors(len(lines2))
        
        for l,c in zip(lines1,colors):
            self.draw(img2, l, c)
            
        for p,c in zip(CF.imagePoints[1][5:11],colors):
            p = p[0]
            cv2.circle(img2,(p[0],p[1]),15,c,-1)
            
        
            
        f = self.writepath+ 'calibration/%d_epo_2.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, img2)
        
        img1 = CF.img1.copy()
        
        for l,c in zip(lines2,colors):
            self.draw(img1, l, c)
            
        for p,c in zip(CF.imagePoints[0][5:11],colors):
            p = p[0]
            cv2.circle(img1,(p[0],p[1]),15,c,-1)
            
        f = self.writepath+ 'calibration/%d_epo_1.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, img1)
        
    def draw(self,img,line,color):
        w = img.shape[1]
        r = line
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [w, -(r[2]+r[0]*w)/r[1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color,1)
        
        return img
        
    def find_subcorners(self,dst,corners,img):
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
            
        f=  self.writepath+'7_subpix.jpg'
        cv2.imwrite(f,img2)
        img2 = transform.rescale(img2, 0.25)
        cv2.imshow('iii',img2)
        cv2.waitKey()
        cv2.destroyWindow('iii')
        
    def find_corners(self,folder, i):
        
        self.folder = folder
        self.i = i
        self.writepath = '../img/results/automated/%d/' % folder
        
        factor = 1
        filename = self.writepath+'calibration/src/%d/1.jpg' % (self.i)
        scene = self.loadImage(filename, factor)
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
        
        f = self.writepath+ '%d_harris.jpg' % (self.i)
        print 'savaing to ' + f
        cv2.imwrite(f, bb)
        
#         self.find_subcorners(harrisdst, corners, scene.view)
        
        return corners,z
        
    def find_objects(self,img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        squares = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs = 150
        retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
        size = 100
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        bin = cv2.dilate(bin,cross)
        bin = cv2.erode(bin,cross)
        
        lf = labeling.LabelFactory([])
        labelsMap = lf.getLabelsExternal(bin, 8, 0)
        return labelsMap
    
    def rrun(self,folder,i):
        self.writepath = '../img/results/automated/%d/' % folder
        #normalnie potrzebne, ale ja czytam z cache'u
#         corners,z = self.find_corners(folder, i)
#         print 'corners: ',map(tuple,corners.tolist())
        
#         offsetX,offsetY,width,height = (z.offsetX,z.offsetY,z.width,z.height)
        offsetX,offsetY,width,height = (1618, 63, 979, 910)
#         offsetX,offsetY,width,height = (2241, 1425, 1630, 1455)
#         corners = [(182, 2348), (195, 2298), (214, 2362), (227, 2314), (239, 2266), (252, 2380), (261, 2329), (273, 2280), (285, 2395), (287, 2231), (296, 2345), (307, 2295), (318, 2246), (323, 2414), (329, 2200), (333, 2362), (343, 2311), (353, 2261), (358, 2432), (364, 2214), (369, 2379), (376, 1740), (376, 2168), (380, 2327), (390, 2278), (394, 1699), (399, 2229), (399, 2451), (406, 2396), (408, 2182), (412, 1749), (415, 1842), (416, 2344), (418, 2139), (425, 2293), (430, 1709), (430, 1800), (435, 2245), (436, 2470), (443, 2197), (445, 2414), (450, 1760), (450, 1851), (452, 1940), (452, 2152), (454, 2361), (463, 2107), (463, 2310), (467, 2061), (468, 1900), (469, 1812), (471, 1722), (472, 2260), (478, 2491), (480, 2212), (484, 2433), (486, 1863), (486, 1949), (488, 1773), (488, 2033), (488, 2166), (493, 2379), (496, 2121), (501, 2327), (503, 2073), (505, 1734), (505, 1913), (505, 1997), (507, 1825), (508, 2276), (516, 2511), (517, 2228), (522, 2044), (523, 1963), (524, 2180), (524, 2452), (525, 1876), (526, 1786), (531, 2135), (532, 2397), (537, 2088), (539, 2344), (541, 2011), (543, 1926), (545, 1838), (547, 2293), (548, 2596), (549, 1746), (554, 2244), (558, 2058), (561, 1976), (561, 2195), (564, 1889), (565, 2473), (566, 1798), (568, 2149), (572, 2415), (575, 2102), (578, 2024), (579, 2361), (582, 1940), (584, 1850), (586, 2310), (593, 2260), (596, 2072), (599, 1989), (599, 2212), (603, 1902), (605, 2164), (606, 1811), (607, 2491), (609, 2117), (613, 2435), (616, 2039), (619, 2380), (620, 1954), (624, 1864), (625, 2327), (630, 1773), (632, 2276), (633, 2086), (637, 2004), (637, 2227), (642, 1916), (643, 2179), (646, 1824), (649, 2131), (653, 2454), (654, 2053), (660, 1968), (660, 2399), (665, 1878), (666, 2345), (668, 1785), (671, 2101), (671, 2293), (676, 2018), (676, 2243), (681, 2195), (682, 1930), (685, 2147), (688, 1838), (693, 2067), (699, 1982), (702, 2416), (706, 1892), (707, 2363), (709, 2116), (712, 2311), (714, 1799), (716, 2033), (716, 2260), (721, 2211), (724, 1945), (725, 2162), (730, 1852), (733, 2083), (741, 1997), (747, 2382), (748, 1906), (749, 2132), (752, 1813), (754, 2330), (757, 2278), (758, 2048), (760, 2227), (763, 2178), (765, 1959), (772, 1867), (774, 2098), (782, 2011), (790, 1921), (790, 2147), (796, 2348), (799, 2063), (799, 2296), (801, 2244), (805, 2193), (807, 1975), (814, 1883), (815, 2113), (824, 2027), (830, 2163), (831, 1934), (838, 2314), (840, 2079), (843, 2262), (844, 2211), (849, 1991), (856, 2129), (865, 2041), (872, 2179), (882, 2097), (885, 2278), (887, 2226), (896, 2144), (913, 2197)]
#         corners = [(1599, 3113), (1600, 3209), (1654, 3065), (1655, 3157), (1656, 3253), (1711, 3016), (1712, 3107), (1714, 3202), (1720, 3302), (1763, 2972), (1767, 3058), (1771, 3150), (1775, 3248), (1779, 3349), (1819, 2927), (1821, 3011), (1826, 3101), (1832, 3196), (1838, 3295), (1846, 3400), (1869, 2884), (1875, 2966), (1881, 3053), (1887, 3145), (1894, 3242), (1901, 3344), (1908, 3449), (1914, 2755), (1922, 2838), (1927, 2920), (1934, 3006), (1942, 3096), (1950, 3190), (1958, 3290), (1964, 2408), (1966, 2464), (1967, 3395), (1968, 2526), (1969, 2580), (1970, 2638), (1971, 2689), (1971, 2794), (1972, 2745), (1978, 2876), (1978, 3504), (1986, 2960), (1995, 3047), (2004, 3140), (2014, 3237), (2024, 3339), (2027, 2441), (2028, 2382), (2029, 2501), (2029, 2560), (2029, 2617), (2030, 2672), (2030, 2727), (2030, 2779), (2034, 3446), (2037, 2914), (2044, 3557), (2047, 3000), (2057, 3090), (2068, 3185), (2079, 3284), (2089, 2868), (2090, 2817), (2090, 3389), (2091, 2764), (2093, 2415), (2093, 2597), (2093, 2653), (2093, 2709), (2094, 2477), (2094, 2538), (2098, 2954), (2103, 3499), (2109, 3042), (2117, 3616), (2121, 3135), (2133, 3231), (2146, 3333), (2152, 2856), (2154, 2802), (2156, 2748), (2157, 2692), (2159, 2635), (2159, 3441), (2160, 2576), (2161, 2514), (2161, 2994), (2162, 3097), (2164, 2456), (2173, 3085), (2173, 3554), (2186, 3179), (2187, 3672), (2200, 3279), (2213, 2946), (2214, 3384), (2215, 2897), (2217, 2843), (2220, 2788), (2223, 2732), (2224, 3036), (2225, 2674), (2228, 2615), (2230, 2554), (2230, 3494), (2231, 2492), (2239, 3129), (2246, 3611), (2253, 3226), (2269, 3328), (2279, 2937), (2283, 2884), (2285, 3435), (2286, 2829), (2289, 3079), (2290, 2772), (2293, 2714), (2297, 2655), (2301, 2593), (2302, 3549), (2305, 3174), (2306, 2533), (2319, 3667), (2322, 3274), (2339, 3378), (2342, 3028), (2346, 2978), (2350, 2926), (2354, 2871), (2356, 3122), (2357, 3489), (2359, 2814), (2364, 2755), (2369, 2696), (2373, 2635), (2374, 3220), (2374, 3606), (2376, 2572), (2392, 3322), (2411, 3430), (2413, 3021), (2418, 2968), (2424, 2913), (2424, 3168), (2430, 2856), (2431, 3541), (2436, 2798), (2441, 2739), (2444, 3268), (2446, 2678), (2454, 2617), (2464, 3373), (2477, 3115), (2483, 3065), (2483, 3483), (2489, 3012), (2494, 3215), (2496, 2956), (2502, 2900), (2509, 2842), (2516, 2782), (2516, 3317), (2523, 2720), (2528, 2657), (2538, 3422), (2553, 3110), (2560, 3057), (2566, 3263), (2567, 3003), (2576, 2945), (2584, 2887), (2587, 3368), (2593, 2826), (2601, 2764), (2612, 2702), (2618, 3205), (2626, 3155), (2634, 3103), (2639, 3310), (2643, 3048), (2652, 2992), (2661, 2933), (2670, 2873), (2680, 2811), (2688, 2748), (2697, 3205), (2706, 3149), (2715, 3098), (2727, 3037), (2737, 2983), (2748, 2919), (2758, 2861)]
        corners = [(183, 2347), (196, 2297), (214, 2362), (227, 2313), (239, 2266), (252, 2380), (261, 2329), (273, 2280), (284, 2395), (288, 2231), (296, 2345), (307, 2295), (318, 2246), (324, 2414), (329, 2201), (332, 2362), (343, 2311), (353, 2261), (358, 2431), (363, 2214), (369, 2379), (376, 1741), (377, 2168), (379, 2327), (390, 2277), (394, 1788), (395, 1699), (399, 2229), (400, 2451), (407, 2396), (408, 2182), (412, 1749), (415, 1842), (416, 2344), (417, 2139), (425, 2293), (430, 1709), (430, 1800), (432, 1887), (435, 2245), (436, 2470), (444, 2197), (445, 2415), (450, 1760), (450, 1851), (452, 1940), (452, 2152), (454, 2361), (463, 2310), (464, 2107), (466, 2060), (468, 1900), (469, 1812), (469, 1984), (472, 1722), (472, 2260), (479, 2491), (480, 2212), (484, 2433), (486, 1863), (486, 1949), (488, 1773), (488, 2166), (489, 2034), (492, 2379), (496, 2121), (501, 2327), (503, 2073), (504, 1996), (505, 1734), (505, 1913), (507, 1825), (509, 2276), (516, 2228), (516, 2511), (522, 1963), (522, 2044), (524, 2181), (524, 2452), (525, 1785), (525, 1876), (531, 2135), (532, 2396), (537, 2088), (539, 2344), (541, 2011), (543, 1926), (545, 1838), (547, 2293), (548, 2596), (549, 1747), (554, 2243), (558, 2058), (561, 1976), (561, 2195), (564, 1889), (565, 2473), (566, 1798), (568, 2149), (572, 2415), (575, 2102), (578, 2024), (579, 2361), (582, 1940), (584, 1851), (585, 1758), (585, 2310), (592, 2260), (595, 2072), (598, 2212), (599, 1989), (603, 1902), (604, 2164), (606, 1811), (607, 2491), (609, 2117), (613, 2435), (616, 2038), (619, 2380), (620, 1953), (624, 1864), (626, 2327), (630, 1773), (631, 2276), (633, 2086), (637, 2004), (637, 2227), (642, 1916), (643, 2179), (646, 1824), (648, 2131), (652, 2454), (654, 2053), (660, 1968), (661, 2398), (665, 1878), (666, 2345), (668, 1785), (671, 2101), (672, 2293), (676, 2018), (677, 2243), (681, 2195), (682, 1930), (685, 2147), (688, 1838), (693, 2068), (699, 1982), (702, 2416), (706, 1892), (707, 2363), (709, 2116), (712, 2311), (715, 1800), (716, 2033), (717, 2260), (721, 2211), (724, 1945), (725, 2162), (730, 1852), (732, 2082), (741, 1997), (746, 2382), (748, 1906), (749, 2132), (752, 1813), (754, 2330), (756, 2278), (758, 2047), (760, 2228), (762, 2178), (765, 1959), (771, 1867), (774, 2098), (783, 2011), (790, 1921), (790, 2146), (796, 2347), (799, 2063), (799, 2296), (801, 2244), (804, 2194), (807, 1975), (814, 1883), (815, 2113), (824, 2027), (830, 1934), (830, 2163), (838, 2314), (840, 2078), (842, 2262), (843, 2211), (849, 1991), (856, 2129), (864, 2040), (872, 2179), (881, 2096), (884, 2278), (886, 2226), (896, 2143), (913, 2197), (927, 2243)]
#         corners = np.array()
        board_size = (20,14)
        
        f = self.writepath+ '%d_zona.jpg' % (i)
        print 'savaing to ' + f
        print 'reading from ' + f
#         cv2.imwrite(f,z.image)
        z = cv2.imread(f,flags=cv2.CV_LOAD_IMAGE_UNCHANGED)
        gray = cv2.imread(f,flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)

        openZ = z.copy()
        
        rho = 1
        theta = np.pi/180
        threshold = 5
        
        mask = np.zeros((z.shape[0],z.shape[1]),np.uint8)
        corners2Shifted = []
        for c in corners:
            c = ( c[0] - offsetY, c[1]-offsetX )
            corners2Shifted.append(c)
            mask[c[0],c[1]] = 1
        mask_size = 3   
        
        mask2 = z.copy()
        img = np.zeros((13,13))
        row,col = draw.circle_perimeter(5,5,3)
        img[row,col] = 1
                
        ret, binary = cv2.threshold(gray,150,1,cv2.THRESH_BINARY_INV)
        b2 = np.zeros(z.shape)
        
        #wykrywanie zgiecia
        
        #erozja
        size = 5
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        b3 = cv2.erode(binary,cross)
        b2[binary==1]=(255,255,255)
        #policzenie powierzchni
        labelMap = measure.label(b3,8,0)
        properties = measure.regionprops(labelMap)
        
        areas = [prop['convex_area'] for prop in properties]
        
        board_w = 10
        board_h = 7
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
                b2[labelMap==p['label']]=(255,0,255)
                edgeBinaryMap[labelMap==p['label']]=1
            if area in backgroundFields:
                b2[labelMap==p['label']]=(0,255,255)
        
        size = 15
        cross = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        edgeBinaryMap = cv2.dilate(edgeBinaryMap,cross)
        
        r = 20
        angles = np.linspace(0, 2*np.pi, 73)
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
                    print binary.shape[1], x
                    print binary.shape[0], y
                    
            edges = np.abs(np.diff(np.array(values,np.float16)))
            ec = int(np.sum(edges))
            
#             cv2.line(mask2,(p[1],p[0]),(y,x),(255,255,2),3)
            row,col = draw.circle_perimeter(p[0],p[1],20)
            if row.max() > b2.shape[0]:
                continue
            if col.max() > b2.shape[1]:
                continue
            if ec ==2:
                b2[row,col] = (0,255,255)
                if edgeBinaryMap[p[0],p[1]] == 1:
                    b2[row,col] = (0,0,255)
                    edgePoints.append(p)
            elif ec ==4:
                b2[row,col] = (255,0,0)
                chessCorners.append(p)
                counter +=1
            elif ec ==0:
                b2[row,col] = (0,0,255)
                pass
            else:
                b2[row,col] = (255,255,0) 
        print 'points:',counter
        print 'shoud be', 10*7*2        
        
        line = measure.LineModel()
        line.estimate(np.array(edgePoints))
        xMax = binary.shape[0]-1
        p1Y = int(line.predict_y(0))
        p2Y = int(line.predict_y(xMax))
        p1 = (p1Y,0)
        p2 = (p2Y,xMax)
        
        cv2.line(b2,p1,p2,(0,0,255),3)
        
        left = np.zeros(binary.shape)
        left[p1[1],p1[0]] = 1
        left[p2[1],p2[0]] = 1
        left[0,0] = 1
        left[xMax,0] = 1
        leftMask = morphology.convex_hull_image(left)
        
        #znajdź wierzchołki opencv
        self.find_corners_opencv(leftMask,openZ,edgeBinaryMap)
        
        cornersLeft = []
        cornersRight = []
        maskLeft = np.zeros(binary.shape,np.uint8)
        maskRight = np.zeros(binary.shape,np.uint8)
        for p in chessCorners:
            if leftMask[p[0],p[1]]:
                cornersLeft.append(p)
                maskLeft[p[0],p[1]] = 1
            else:
                cornersRight.append(p)
                maskRight[p[0],p[1]] = 1
            
        rho = 3
        theta = np.pi/360
        threshold = board_h-2
        
        lines = cv2.HoughLines(maskRight,rho,theta,threshold)
        lines2 = lines[0]
        lines2 = self.find_main_directions_lines(lines2)
        if lines2 is not None:
            thetas = []
            for (rho, theta) in lines2:
                thetas.append(theta)
            thetas.sort()
                
#             mark.drawHoughLines(lines2, b2, (0,255,0), 4)
        
        leftPoints  = self.getImagePoints(cornersLeft, b2.shape,True)
        rightPoints = self.getImagePoints(cornersRight,b2.shape,False)
        finalPoints, finalWorldPoints = self.getWorldPoints(leftPoints,rightPoints)
        
        print finalPoints.shape
        print finalPoints.tolist()
        print finalWorldPoints.shape
        print finalWorldPoints.tolist()
        
        
        print 'left', leftPoints
        print 'right', rightPoints
        
        wiersz = 4
        kolumna = 9
#         kk = (leftPoints[wiersz][kolumna][1],leftPoints[wiersz][kolumna][0])
#         cv2.circle(b2,kk,5,(0,0,255),-1)
        
#         kk = (rightPoints[wiersz][kolumna][1],rightPoints[wiersz][kolumna][0])
#         cv2.circle(b2,kk,5,(0,0,255),-1)
                
        cv2.imshow('iii',b2)
        cv2.waitKey()
        cv2.destroyWindow('iii')
        f = self.writepath+ '%d_corners%d.jpg' % (7,2)
        cv2.imwrite(f,b2)    
        mask2[mask==1] = (0,255,0)

    def find_corners_opencv(self,leftMask,origin,edgeBinaryMap):
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
    
    def orderPoints(self,points):
        orderedPoints = []
        py = [p[0] for p in points]
        m = max(py)
        while min(py)<=m:
            index = py.index(min(py))
            orderedPoints.append( points[ index ])
            py[index] = m+1 

        return orderedPoints
    
    def getImagePoints(self,points,shape,left=True):
        orderedPoints = []
        
        newPoints = list(points)
        while len(newPoints)>0:
            scanLinePoints = self.getScanLine(points,shape,left)
            p = self.orderPoints(scanLinePoints)
            orderedPoints.append( p )
            newPoints = self.removePointsFromList(points, scanLinePoints)
            
        return orderedPoints
    
    def getWorldPoints(self,leftPoints,rightPoints):
        finalPointsL = np.zeros( (len(leftPoints),len(leftPoints[0]),3) )
        finalPointsR = np.zeros( (len(leftPoints),len(leftPoints[0]),3) )
        
        for row in range(len(leftPoints)):
            for col in range(len(leftPoints[row])):
                p =  (row,col,0)
                finalPointsL[row,col] = p
                
        for row in range(len(rightPoints)):    
            for col in range(len(leftPoints[row])):
                p =  (len(leftPoints),col,len(rightPoints)-(row))
                finalPointsR[row,col] = p
        
        finalPoints = np.append(finalPointsL,finalPointsR,0)
        final = np.append(leftPoints,rightPoints,0)

        return final, finalPoints 
    
    
    def find_main_directions_lines(self,lines,linesExtreme='max'):
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
        
        
    def test_10_3(self):
        self.rrun(10,7)
        
#     def test_10_4(self):
#         self.rrun(10,4)   
        
       
       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    