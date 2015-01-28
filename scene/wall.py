# -*- coding: utf-8 -*-
'''
Created on Jan 8, 2015

@author: Tomasz
'''
import cv2
import numpy as np
import func.analise as an
from func import objects as obj
from math import cos
import func.markElements as mark
from calculations.labeling import LabelFactory

class Wall(object):
    '''
    classdocs
    '''


    def __init__(self,label,wallMap,area2):
        '''
        Constructor
        '''
        
        self.map = wallMap
        self.label = label
        
        self.shadow = self.isShadow()
        
        #kontur sciany        
        cnts = cv2.findContours(wallMap,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        cnt = cnts[0][0]
        self.cnt = cnt
        
        #obrys scany
        self.convex = None
        hull = cv2.convexHull(cnt,returnPoints = False)
        self.convexHull = hull
        
        #defect obrysu
        defs = cv2.convexityDefects(cnt,hull)
        self.hullDefects = defs
        self.convex = self.analyzeDefects(defs,cnt)
        self.convex_point = None
        self.conterpoint = None
        
        # map of the distances from the wall
#         wallInverted = np.where(labelsMap == label ,0,1).astype('uint8')
        wallInverted = area2#np.where(wallMap == 1 ,0,1).astype('uint8')
        self.wallDistance = cv2.distanceTransform(wallInverted,cv2.cv.CV_DIST_L1,3)
        
        self.contours = []
        self.contoursDict = {}
        self.nodes = []
        
        self.vertexes = []
        
        self.lines = []
        
        
    def analyzeDefects(self,defs,cnt,treshold = 20):
        fars = []
        lines = []
        for kk, defects in enumerate(defs): 
                
                for jj in range(defects.shape[0]):
                    s,e,f,d = defects[jj]
                    
                    #line endings
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    
                    #defect point the furthest from the line
                    far = tuple(cnt[f][0])
                    line = an.getLine(start, end,0)
                    
                    distance = an.calcDistFromLine(far, line)
                    
                    if distance > treshold:
                        print 'wielobok wypukly. Punkt:',far
                        fars.append(far)
                        lines.append((start, end))
        if len(fars)>0:
            return (True,fars,lines)
        else:
            return (False,fars,lines)
    
    
    def findPotentialCorners(self):
        points = []
        corners = []
        for x in self.cnt:
            point = tuple(x[0])
            points.append(point)
        indexes = self.__findCornersOnContour(points, 50)
        for ii in indexes:
            corners.append(points[ii])
            
        return corners

    
    def __findCornersOnContour(self,contour,size):
        '''
            size - dlugosc probierza, offset pomiedzy elementami konturu dla ktorych zrobiony jest odcinek
        '''
    
        if len(contour)>size:
            indexes = []
            dist = an.calcDistances(contour,size,int(size*0.1))
    
            for d in dist.iterkeys():
                segment1 = dist[d]
                MaxValue = max(np.asarray(segment1)[:,1])
                index = np.where(segment1 == MaxValue)[0]
                indexes.append(segment1[index[0]][0])
            return indexes
        else:
            return []
        
    def __getLines(self):
        '''
            sumuje linie nalezace do poszczegolnych konturow
        '''
        lines = []
        contours = self.contours
        contoursMap = np.zeros_like(self.map)
        for c in contours:
            linesC = c.getLines()
            if c.wayPoint is not None:
                lines.extend(linesC)
                contoursMap[c.map==1] = 1
            else:
                lines.extend(linesC[:1])
            
        #eliminacja prawie identycznych lini rozniacych sie o niewielki kat i wynikajacych z tolerancji lamanej
        for n,l1 in enumerate(lines):
            if l1 is None:
                continue
            for m,l2 in enumerate(lines):
                if l2 is None:
                    continue 
                if l1[0] == l2[0] and l1[1] == l2[1]:
                    break
                x = cos(abs(l1[1]-l2[1]))
                if x > 0.95:
                    shape = self.map.shape
                    a1,b1,c1 = an.convertLineToGeneralForm((l1[0], l1[1]), shape)
                    a2,b2,c2 = an.convertLineToGeneralForm((l2[0], l2[1]), shape)
                    p = an.get2LinesCrossing((a1, b1, c1), (a2, b2, c2))
                    if p != False:
                        px,py = p
                        #przeciecie na obrazie
                        if 0 < px and px < shape[1] and 0 < py and py < shape[1]:
                            # znajdz która jest dokladniejsza
                            int1 = self.__getFitMeasure(l1, contoursMap)
                            int2 = self.__getFitMeasure(l2, contoursMap)
                            if int1 > int2:
                                lines[m] = None
                            else:
                                lines[n] = None
                    elif x == 1 and abs(l1[0]-l2[0]) < 5:
                        int1 = self.__getFitMeasure(l1, contoursMap)
                        int2 = self.__getFitMeasure(l2, contoursMap)
                        if int1 > int2:
                            lines[m] = None
                        else:
                            lines[n] = None
                        
        #usuwanie None z listy lini
        finalLines = []
        for l in lines:
            if l == None:
                continue
            finalLines.append(l)
        self.lines = finalLines
        
        return self.lines    

    def __getFitMeasure(self,line,contoursMap):
        '''
            mierzy jak bardzo dana linia pokrywa sie z konturem
        '''
        lineMap1 = np.zeros_like(self.map)
        mark.drawHoughLines([line], lineMap1, 1, 2)
        coverage1 = cv2.bitwise_and(lineMap1,contoursMap)
        int1 = np.sum(coverage1)
        
        return int1
        
        
    def getLinesCrossings(self):
        '''
        zwraca punkty bedące przecięciami podanych prostych
    
        lines - lista prostych będących krotkami [ (a,b,c) , ... ]
        edge - płótno
    
        return [ (x,y) ] - lista puntków będacych przecieciami
        '''
        lines = []
        shape = self.map.shape
        boundaries = np.array([[0,0],[shape[1],0],[shape[1],shape[0]],[0,shape[0]]])
        
        lines = self.__getLines()
        linesGeneral = []
        generalDict = {}
        
        for (rho, theta) in lines:
            # blue for infinite lines (only draw the 5 strongest)
            a,b,c = an.convertLineToGeneralForm((rho,theta),shape)
            linesGeneral.append((a,b,c))
            generalDict[(a,b,c)] = (rho,theta)
            
        
        crossing = []
        property = {}
        fars = self.convex
        
        for k in linesGeneral:
            for l in linesGeneral:
                if k == l:
                    continue
                
                p = an.get2LinesCrossing(k,l)
                # sprawdź czy leży wewnątrz strefy
                if p != False:
                    if p in crossing:
                        continue
                    isinside = cv2.pointPolygonTest(boundaries,p,0)
                    if isinside>0:
                        dist = self.wallDistance[p[1],p[0]]
                        if dist<50:
                            if p != False:                                
                                crossing.append(p)
                                property[p] = []
                                property[p].append(k)
                                property[p].append(l)
                    else:
                        pass
        #dla wklesłych
        if self.convex[0]:
            point = self.__findClosestCrossingConvex(crossing)
            
            #eliminacja punktow od wkleslosci
            img = np.zeros_like(self.map)    
            # linie przeciete mapa sciany
            for i,line in enumerate(property[point]):
                l = generalDict[(line[0],line[1],line[2])]
                
                mark.drawHoughLines([l], img, 1, 2)
                img = cv2.subtract(img,self.map)
            
            # etykietowanie kawalkow lini    
            lf = LabelFactory([])
            res = lf.getLabelsExternal(img, neighbors=8, background=0)
            labelOK = np.unique(res[(point[1],point[0])])[0]
            Labels =  np.unique(res)
            
            crossMask = np.zeros_like(self.map)  
            #wszystko co nie jest polaczone z punktem wkleslosci nalezy do maski
            for label in Labels:
                if label == -1 or label == labelOK:
                    continue
                indexes = np.where(res == label)
                crossMask[indexes] = 1
            
            #usun przeciecia pokrywajace sie z maska
            for i,v in enumerate(crossing):
                if crossMask[(v[1],v[0])] == 1:
                    crossing[i] = None
#                     crossing.remove(crossing[i])
            crossing2 = []
            for c in crossing:
                if c is not None:
                    crossing2.append(c)
            crossing = crossing2
#         print 'crossings :', crossing
        return crossing, fars#,poly,vertexes
    
    def getVertexes(self,crossings,farlines=(False,0,0)):
        '''
        zwraca uporzadkowana liste wierzcholkow
        
        bez srodkow punktow potrojnych -> wciaz z bledami
        '''
        vertexes = []
        if len(crossings)>0:
            points = np.array([crossings])
            polygonG = cv2.convexHull(points,returnPoints = True)
            polygonG2 = map(tuple,polygonG)
            #dla wklesłych
            if self.convex[0]:
                (start, end) = self.convex[2][0]
                cross_defect = self.__findClosestCrossingConvex(crossings)
                min1 = 1000
                min2 = 1000
                
                #kazdy punkt obwiedni dodaj do wierzchołków
                #i dla lini defektowej znajdź punkty najbliższe jej startu i końca 
                for p in polygonG2:
                    p = (p[0][0],p[0][1])
                    vertexes.append(p)
                    
                    d1 = an.calcLength(p, start)
                    d2 = an.calcLength(p, end)
                    if d1<min1:
                        min1 = d1
                        s = p
                    if d2<min2:
                        min2 = d2
                        e = p
                idx1 = vertexes.index(s)
                idx2 = vertexes.index(e)
                # indexy beda kolejno po sobie chba ze akurat beda na brzegach listy
                if idx2-idx1 == 1:
                    vertexes.insert(idx2, cross_defect)
                else:
                    vertexes.insert(idx1, cross_defect)
                self.convex_point = cross_defect
                    
            #dla wypukłych
            else:        
                for p in polygonG2:
                    p = (p[0][0],p[0][1])
                    vertexes.append(p)
                    
            
            
            #vertexy sa uporzadkowane wiec mozna wyeliminowac te punkty ktore sa nadal na jednej lini
#             lines = self.lines
#             for line in lines:
#                 img = np.zeros_like(self.map)
#                 mark.drawHoughLines([line], img, 1, 2)
#                 values = []
#                 for i,v in enumerate(vertexes):
#                     if v == self.convex_point:
#                         
#                         values.append(0)
#                     else:
#                         values.append(img[v[1],v[0]])
#                 
#                 if sum(values) > 2:
#                     if values[0] == 1 and values[-1] == 1 and values[2] == 1:
# #                         vertexes.remove(vertexes[0])
#                         pass
#                     if values[0] == 1 and values[-1] == 1 and values[-2] == 1:
# #                         vertexes.remove(vertexes[-1])
#                         pass
#                     else:
#                         index =  values.index(1)+1
#                         vertexes.remove(vertexes[index])
        self.vertexes = vertexes
#         print 'vertex',vertexes
        return vertexes
    
    def __findClosestCrossingConvex(self,crossings):
        if self.convex[0]:
            point = self.convex[1][0]
            #znajdź przeciecie najblizsze temu punktowi
            min0 = 1000
            for cross in crossings:
                d0 = an.calcLength(point, cross)
                if d0<min0:
                    min0 = d0
                    cross_defect = cross
            return cross_defect
        else:
            return False
        
    def isShadow(self):
        wMap = self.map
        bottom = wMap[-4,:]
        nz = np.nonzero(bottom)
        siz = len(nz[0])
        if siz > 0:
            print 'cien'
            return True
        else:
            return False