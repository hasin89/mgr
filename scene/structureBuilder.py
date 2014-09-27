# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
from Polyline import Polyline
from Segment import Segment
from Vertex import Vertex
import cv2
import copy
import analyticGeometry as ag
from LineDectecting import LineDetector
import numpy as np

class StructureBuilder(object):
    
    def __init__(self,shape):
        
        self.shape = shape        
    
    def getMostLeftAndRightCorner(self,corners):
        '''
        zwraca wierchołki o najmniejszej i największej współrzędnej X
    
        corners - ndarray - wierzchołki (x,y)
        shape - tuple (h,w)
        '''
        shape = self.shape
    
        # na wypadek gdyby nie znalazły się żadne wierzchołki wewnątrz głównego konturu
        if corners.size > 0:
            #znajdz wierzchołek o naniejszej wspolrzednej X
            leftX = min(corners.T[0])
            leftIndex = corners.T.tolist()[0].index(leftX)
            leftY = corners.T[1][leftIndex]
            left = (leftX,leftY)
    
            #znajdz wierzcholek o najwiekszej wspolrzednj X
            rightX = max(corners.T[0])
            rightIndex = corners.T.tolist()[0].index(rightX)
            rightY = corners.T[1][rightIndex]
            right = (rightX,rightY)
        else:
            left = (0,0)
            right = (shape[1],shape[0])
    
        return left,right
    
    
    def getAllCrossings(self,lines,boundaries):
        '''
        zwraca punkty bedące przecięciami podanych prostych
    
        lines - lista prostych będących krotkami [ (a,b,c) , ... ]
        edge - płótno
    
        return [ (x,y) ] - lista puntków będacych przecieciami
        '''
        
        shape = self.shape
        linesGeneral = []
    
        for (rho, theta) in lines:
            # blue for infinite lines (only draw the 5 strongest)
            a,b,c = ag.convertLineToGeneralForm((rho,theta),shape)
            linesGeneral.append((a,b,c))
    
        pairs = [(linesGeneral[i],linesGeneral[i+1]) for i in range(0,len(linesGeneral)-1)]
        pairs.append((linesGeneral[-1],linesGeneral[0]))
    
        pairs.extend([(linesGeneral[i],linesGeneral[i+2]) for i in range(0,len(linesGeneral)-2)])
    
        # pairs.extend([(linesGeneral[i],linesGeneral[i+3]) for i in range(0,len(linesGeneral)-3)])
    
    
        segments = {}
        vertexes = {}
    
        for i in range(0,len(linesGeneral)):
            segments[linesGeneral[i]] = Segment()
            segments[linesGeneral[i]].line = linesGeneral[i]
    
        crossing = []
        good = 0
        j= 0
    
        #znajdź właściwe przecięcia
        #todo budowa wierzchołków strukturalnych
        for i,(k,l) in enumerate(pairs):
    
            p = ag.get2LinesCrossing(k,l)
            # sprawdź czy leży wewnątrz ramy
            if p != False:
                isinside = cv2.pointPolygonTest(boundaries,p,0)
                if isinside>0:
                    if p != False:
                        crossing.append(p)
    
                        s1 = segments[k]
                        s1.neibourLines.append(l)
                        s1.points.append(p)
                        segments[k] = s1
    
                        s2 = segments[l]
                        s2.neibourLines.append(k)
                        s2.points.append(p)
                        segments[l] = s2
    
                        vertex = Vertex(p)
                        vertex.lines.append(l)
                        vertex.lines.append(k)
                        vertexes[p] = vertex
    
                        good += 1
    
                else:
                    pass
            if good == len(linesGeneral):
                break
    
        segmentsList = segments.values()
        poly = Polyline()
    
        for s in segments.values():
            if len(s.points) > 1:
                vertexes[s.points[0]].neibours[s.line] = s.points[1]
                vertexes[s.points[1]].neibours[s.line] = s.points[0]
    
    
        #  nie wiem czy to do czegoś potrzbne jest
        flag = [True for true in segmentsList]
        if (len(segmentsList[0].points)) > 1 & flag[0]:
            poly.segments[0] = segmentsList[0]
            poly.begining = segmentsList[0].points[0]
            poly.ending = segmentsList[0].points[1]
            poly.points.append(segmentsList[0].points[0])
            poly.points.append(segmentsList[0].points[1])
            flag[0] = False
    
            i = 1
            while(i<len(linesGeneral)):
                s = segmentsList[i]
                points = list(segmentsList[i].points)
                if (poly.ending in points) & flag[i] & (len(segmentsList[i].points) > 1):
                    index = points.index(poly.ending)
                    if index == 0:
                        index = 1
                    else:
                        index = 0
                    p = s.points[index]
                    poly.ending = p
                    poly.points.append(p)
                    poly.segments[len(poly.segments)] = segmentsList[i]
                    flag[i] = False
                    i=0
                else:
                    i += 1
    
        return crossing,poly,vertexes


    def getInnerSegments(self,otherLines,poly):
        '''
        szukanie odcinka będącego krawędzią wewnętrzeną
    
        '''
        
        shape = self.shape
        result = {}
        for l in otherLines:
            l = ag.convertLineToGeneralForm(l,shape)
            min = 100000
            points = list(poly.points)
            del points[-1]
            for p in points:
                dist = ag.calcDistFromLine(p,l)
                if dist < min:
                    min = dist
                    Pmin1 = p
            del points[points.index(Pmin1)]
    
            #szukanie drugoego punktu
            min = 100000
            for p in points:
                dist = ag.calcDistFromLine(p,l)
                if dist < min:
                    min = dist
                    Pmin2 = p
    
            s = Segment()
            s.setPoints(Pmin1,Pmin2)
            result[l] = s
        # Pmin2 = (0,0)
    
        return result.values()
    
    
    def addSegmentsToStructure(self,innerSegments,vertexes):
        for s in innerSegments:
            for p in s.points:
                vertexes[p].lines.append(s.line)
                index = s.points.index(p)
                if index == 1:
                    vertexes[p].neibours[s.line] = s.points[0]
                else:
                    vertexes[p].neibours[s.line] = s.points[1]
        return vertexes
    
    
    def makeFaces(self,vertexesORG):

        vertexes = copy.deepcopy(vertexesORG)
        line = {}
        i=-1
        for v in vertexes.values():
            start = v.point
            #wszystkie trasy z tego punktu
            routes = []
            #jeśli sa jeszcze jacyś sąsiedzi startu
            while len(vertexes[start].neibours.values())>0:
                # zacznij spisywać trasę
                route = []
                route.append(start)
                point = start
    
                while len(vertexes[point].neibours.values())>0:
    
                    #pobierz sasiadów
                    neibours = vertexes[point].neibours.values()
    
                    #usuń tych co już byli na tej trasie
                    for r in route:
                        if r in neibours:
    
                            del vertexes[point].neibours[vertexes[point].neibours.index(r)]
    
                    #jeżeli już nikt nie został (zatoczyła się pętla)
                    if len(neibours) == 0:
                        break
    
                    #wybierz następnika
                    point = neibours[0]
                    route.append(point)
    
                    pass
                    #usuń poprzednika
                    # del vertexes[point].neibours[vertexes[point].neibours.values().index(point)]
                routes.append(route)
                pass
    
                # while start not in point.neibours.values():
                #     for l in line:
                #         line[i].append(neibours[i])
                #
                #     neibours = vertexes[point].neibours.values()
                #     if start in neibours:
                #         break
    
    
    
    
    
            pass
        # line[i].append(point)
        # flag = True
        # j=0
        # while(flag):
        #     points = list(vertexes[point].neibours.values())
        #     for l in line[i]:
        #         if l in points:
        #             del points[points.index(l)]
        #     if len(points)>0:
        #         point = points[0]
        #         line[i].append(point)
        #     else:
        #         break
    
    
    
        return line
    
    
    def tryMatch(self,corners,left,right):
        """
            próbuje dopasować prawy i lewy punkt???
        """
        min = left[0]
        max = right[0]
        Xs = []
    
        for c in corners:
            x = c[0]-min
            x /= float(max-min)
            Xs.append((x,(c[0],c[1])))
        Xs.sort()
        return Xs