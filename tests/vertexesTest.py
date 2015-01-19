# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
from calculations import triangulation, DecompPMat, labeling
import cv2
import func.markElements as mark
from scene.mirrorDetector import mirrorDetector
from scene.zone import Zone

from scene.objectDetector2 import objectDetector2

from drawings.Draw import getColors

from calculations.labeling import LabelFactory
from drawings.Draw import getColors

from scene import analyticGeometry
from scene import edgeDetector

from skimage import morphology
from skimage import measure

import func.analise as an
from math import sqrt
from scene.wall import Wall
from scene.qubic import QubicObject
import calculations
from scene.LineDectecting import LineDetector

from math import cos
from collections import Counter
from func.markElements import corners

class VertexesTest(unittest.TestCase):
    
    POINTS = []


    def setUp(self):
        np.set_printoptions(precision=4)

    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def getZones(self,folder,pic):
        factor = 1
        filename = '../img/%d/%d.JPG' % (folder, pic)
        scene = self.loadImage(filename, factor)
        
        md = mirrorDetector(scene)
        md.findMirrorZone()
        
        od = objectDetector2(md,scene.view)
        zoneA,zoneB,zoneC,zoneD = od.detect()
        
        return scene,zoneA,zoneB,zoneC,zoneD
       
    def loadZone(self,folder,pic,letter):
        factor = 1
        filename = '../img/results/automated/%d/objects2/%d_objects_on_mirror_%s.jpg' % (folder,pic,letter)
        zone = self.loadImage(filename, factor)
        if folder ==9 and pic == 1 and letter == 'B':
            self.POINTS = [
                           (2544,691),
                           (2622,624),
                           (2760,671),
                           (2684,740),
                           (2748,806),
                           (2533,827),
                           (2672,877)
                           ]
            
        
        return zone
    
    def eliminateSimilarLines(self,walls):
        """
            eliminuje linie zbyt podobne do siebie
        """
#         w = Wall()
        
        for kk,w in walls.iteritems():
            
            for i1, c1 in enumerate(w.contours):
                deletedLines = []
                lines1 = c1.getLines()
                for line in lines1:
                    for i2, c2 in enumerate(w.contours):
                        if c1.label == c2.label:
                            continue
                        lines2 = c2.getLines()
                        for l in lines2:
                            x = abs(l[1]-line[1])
                            value = cos(x)
                            #TODO: - jeśli znajdzie się para prostych równoległych,
                            # ale odległych o znaczną odległość (równoległe boki) to trzeba temu bedzie zaradzić
                            if value>0.9:
                                print 'similar',value,l,line
                                #JEŚLI SĄ RÓWNOLEGŁE ALE NIE BLISKO SIEBIE TO NIE LICZ ICH
                                shape = w.map.shape
                                lG = an.convertLineToGeneralForm((l[0], l[1]), shape)
                                lineG = an.convertLineToGeneralForm((line[0], line[1]), shape)
                                cross = an.get2LinesCrossing(lG, lineG)
                                if cross != False:
                                    boundaries = np.array([[0,0],[shape[1],0],[shape[1],shape[0]],[0,shape[0]]])
                                    isinside = cv2.pointPolygonTest(boundaries,cross,0)
                                    
                                    if isinside>0:
                                        print 'redundant'
                                        print l
                                        print line
                                        print ''
                                        if len(lines1) > len(lines2):
                                            final = []
                                            linesTMP = walls[kk].contours[i2].getLines()
                                            for ltmp in linesTMP:
                                                if ltmp[0] != l[0] and ltmp[1] != l[1]:
                                                    final.append(ltmp)
                                            walls[kk].contours[i2].lines = np.array(final)
                                            
                                            
                                        else:
                                            final = []
                                            linesTMP = walls[kk].contours[i1].getLines()
                                            for ltmp in linesTMP:
                                                if ltmp[0] != line[0] and ltmp[1] != line[1]:
                                                    final.append(ltmp)
                                            walls[kk].contours[i1].lines = np.array(final)
                for d in deletedLines:
                    if d in lines1:
                        print 'rmove'
        
        return walls
    
    def eliminateSimilarPoints(self,crossings,corners):
        for p1 in crossings:
            min2 = 1000
            for p2 in crossings:
                if p1 != p2:
                    dist = an.calcLength(p1, p2)
                    if dist<min2 :
                        min2 = dist
                        closest = p2
            
            min1 = 1000
            for corn in corners:
                dist1 = an.calcLength(p1, corn)
                if dist1 < min1:
                    min1 = dist1
                    closestcorn = corn
            # z dwoch bliskich sobie zostaw ten blizej potencjlnego rogu
            if min2 < 30:
                dist1 = an.calcLength(p1, closestcorn)
                dist2 = an.calcLength(closest, closestcorn)
                if dist1<dist2:
                    crossings.remove(closest)
                else:
                    crossings.remove(p1)
            # jesli jakis punkt jest dalej niz 50 od potencjalnego rogu to go wymaz
            if min1 > 30:
                crossings.remove(p1)
    
    def getVertexes(self,wall,crossings):
        vertexes = []
        if len(crossings)>0:
            points = np.array([crossings])
            polygonG = cv2.convexHull(points,returnPoints = True)
            polygonG2 = map(tuple,polygonG)
            #dla wklesłych
            if wall.convex[0]:
                (start, end) = wall.convex[2][0]
                point = wall.convex[1][0]
                min1 = 1000
                min2 = 1000
                
                #kazdy punkt obwiedni dodaj do wierzchołków i znajdx punkty najbliższe startu i końca
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
                if idx2-idx1 == 1:
                    vertexes.insert(idx2, point)
                else:
                    vertexes.insert(idx1, point)
                    
#                 for idx range(len(vertexes)):
#                     an.calcDistFromLine(point, line)
                    
#               jesli jest wiecej niż 6 wierzchołków to jest coś nie tak  
#             if len(vertexes)>6:
#                     vectors = []
#                     for n in range(len(vertexes)-1):
#                         vec = (vertexes[n][0]-vertexes[n+1][0],vertexes[n][1]-vertexes[n+1][1])
#                         vectors.append(vec)
#                     vec = (vertexes[-1][0]-vertexes[0][0],vertexes[-1][1]-vertexes[0][1])
#                     vectors.append(vec)
#                     
#                     for j in range(len(vectors)):
#                         if  vectors[j][0] < 5 and  vectors[j][1] < 5:
#                             vertexes.remove(vertexes[j])
#                             flag = True
#                             break
#                             
#                             
#                     print 'vectors',vectors
#                     if not flag:
#                         for j in range(len(vectors)):
#                             if abs(vectors[j][0]) == 0:
#                                 vectors[j] = (0 , vectors[j][1] )
#                             else:
#                                 vectors[j] = (vectors[j][0] / abs(vectors[j][0]) , vectors[j][1] / abs(vectors[j][0]))
#                                 
#                         for i in range(len(vectors)-1):
#                             if vectors[i] == vectors[i+1]:
#                                 vertexes.remove(vertexes[i])
#                                 break
#                         if vectors[-1] == vectors[0]:
#                             vertexes.remove(vertexes[-1])
#                         print 'vectors',vectors
                    
            #dla wypukłych
            else:        
                for p in polygonG2:
                    p = (p[0][0],p[0][1])
                    vertexes.append(p)
            print 'vertex',vertexes
        return vertexes
        
       
    def proceed(self,zone,mask,folder, i, letter):
        pic = i
        
        qubic = QubicObject(zone)
        
        #caly bialy
        image = qubic.emptyImage.copy()
        image2 = image.copy()
        
        image3 = image.copy()
        
        
        
        image5 = image2.copy()
        lf = LabelFactory()
        image5 = lf.colorLabels(image5, qubic.edgeLabelsMap, -1)
        
        f = '../img/results/automated/%d/obj4/%d_objects_on_mirror_%s_skeleton.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
        cv2.imwrite(f, image5)
        
        image1 = image2.copy()
        image1[qubic.edgeMask == 1] = (255,255,255)
        f = '../img/results/automated/%d/obj4/%d_objects_on_mirror_%s_sobel.jpg' % (folder, pic,letter)
        print 'savaing to ' + f
#         cv2.imwrite(f, image1)
        
        image2[qubic.skeleton2 == 1] = (255,255,255)

        
        #znalezienie lini na szkielecie
        rho = 0.5
        theta = np.pi/45
        part = 25
        
        rho = 1
        theta = np.pi/45
        part = 10
        
            
        image2 = lf.colorLabels(image2,qubic.labelsMap)
         
#         self.eliminateSimilarLines(qubic.walls)    
        
        for kk,wall in qubic.walls.iteritems():
            image4 = image2.copy()
            corners = wall.findPotentialCorners()
            for p in corners:
                cv2.circle(image4,p,5,(255,0,255),2)
                
            for p in self.POINTS:
                cv2.circle(image4,p,10,(255,255,0),3)
            
            f = '../img/results/automated/%d/obj4/%d_objects_on_mirror_%s_hull_%d.jpg' % (folder, pic,letter,kk)
            print 'savaing to ' + f
#             cv2.imwrite(f, image4)
            
            image3 = image.copy()
            colors = getColors(len(wall.contours))
            ii = 0
            
            for c in wall.contours:
                
                polygon  = c.polygon
                if len(polygon)>1:
                    for i in range(len(polygon)-1):
                        cv2.line(image3, (polygon[i][0][1],polygon[i][0][0]), (polygon[i+1][0][1],polygon[i+1][0][0]),(255,255,255),1)
                
                ll = map(np.array,np.transpose(np.array(c.points)))
                image3[ll] = (0,255,255)
                c.getLines()
                lines = c.lines
                if lines is not None:
#                     print lines
    
                    #eliminate close to each other lines:
                    ld = LineDetector(zone.view.shape)
                    
                    
                    lines2 = lines
                    
                    if c.wayPoint is not None:
                        mark.drawHoughLines(lines, image3, (128,128,128), 1)
                    else:
                        mark.drawHoughLines(lines[:1], image3, (128,128,255), 1)
                    
                    if c.wayPoint is not None:
                        mark.drawHoughLines(lines2, image3, colors[ii], 2)
                        ii +=1
                    else:
                        mark.drawHoughLines(lines2[:1], image3, (128,255,255), 2)
                        
            image3[wall.map == 1] = (255,255,255)    
#             for p in corners:
#                 cv2.circle(image3,p,5,(255,0,255),2)           
            crossings,fars = wall.getLinesCrossings()
#             for i in range(0,len(mainBND)-1):
#                 mark.drawMain(image3,(mainBND[i][0],mainBND[i][1]) ,(mainBND[i+1][0],mainBND[i+1][1]))

#             crossings = self.eliminateSimilarPoints(crossings,corners)
            
            vertexes = self.getVertexes(wall, crossings)
            
            for vv in vertexes:
                cv2.circle(image3,(vv[0],vv[1]),4,(255,0,0),-1) 
            
            
#             for cc in crossings:
#                 cv2.circle(image3,(cc[0],cc[1]),4,(255,0,0),-1) 
            
            
            print fars
            if fars[0] == True:
                for f in fars[1]:
                    print 'far:',f
#                     cv2.circle(image3,(f[0],f[1]),7,(255,0,0),2)
                      
            for node in wall.nodes:
#                 ll = map(np.array,np.transpose(np.array(c.points)))
                cv2.circle(image3,(node[1],node[0]),2,(255,255,0),-1)
                
            
            
            f = '../img/results/automated/%d/obj4/%d_lines_%s_%d.jpg' % (folder, pic,letter,kk)
            print 'savaing to ' + f
            cv2.imwrite(f, image3)
            
        return image,mask


    def execute(self, folder, pic):
        i = pic
        
        zoneA = self.loadZone(folder, pic,'A')
        zoneB = self.loadZone(folder, pic,'B')
        zoneC = self.loadZone(folder, pic,'C')
        zoneD = self.loadZone(folder, pic,'D')
        
        ed = edgeDetector.edgeDetector(zoneA.view)
        mask = ed.getSobel()
        imgA,maskA = self.proceed(zoneA,mask,folder, i,'A')
        
        ed = edgeDetector.edgeDetector(zoneB.view)
        mask = ed.getSobel()
        imgB,maskB = self.proceed(zoneB,mask,folder, i,'B')
        
        ed = edgeDetector.edgeDetector(zoneC.view)
        mask = ed.getSobel()
        imgC,maskC = self.proceed(zoneC,mask,folder, i,'C')
        
        ed = edgeDetector.edgeDetector(zoneD.view)
        mask = ed.getSobel()
        imgD,maskD = self.proceed(zoneD,mask,folder, i,'D')
        
        
    
#     def test_9_8(self):
#         self.execute(9, 8)
        
    def test_9_1(self):
        self.execute(9, 1)
        
    def test_9_2(self):
        self.execute(9, 2)
# #         
    def test_9_3(self):
        self.execute(9, 3)
            
    def test_9_7(self):
        self.execute(9, 7)
#          
#          
    def test_9_10(self):
        self.execute(9, 10)
         
    def test_9_11(self):
        self.execute(9, 11)           

#     def test_8_3(self):
#         self.execute(8, 3)
#         
#     def test_8_6(self):
#         self.execute(8, 6)   
#         
#     def test_8_7(self):
#         self.execute(8, 7) 
#     
#     def test_8_19(self):
#         self.execute(8, 19) 
# 
#     def test_8_77(self):
#         self.execute(8, 77)

#     def test_7_1(self):
#         self.execute(7, 1)   
#         
#     def test_7_2(self):
#         self.execute(7, 2) 
#     
#     def test_7_3(self):
#         self.execute(7, 3) 
# 
#     def test_4_1(self):
#         self.execute(4, 1)   
#         
#     def test_4_2(self):
#         self.execute(4, 2) 
#     
#     def test_4_3(self):
#         self.execute(4, 3)
#     
#     def test_4_4(self):
#         self.execute(4, 4)   
#         
#     def test_4_5(self):
#         self.execute(4, 5) 
#     
#     def test_4_6(self):
#         self.execute(4, 6)
#     
#     def test_4_7(self):
#         self.execute(4, 7)   
#         
#     def test_4_8(self):
#         self.execute(4, 8) 
#     
#     def test_4_9(self):
#         self.execute(4, 9)  
# 
#     def test_5_1(self):
#         self.execute(5, 1)   
#         
#     def test_5_2(self):
#         self.execute(5, 2) 
#     
#     def test_5_3(self):
#         self.execute(5, 3)
#     
#     def test_5_4(self):
#         self.execute(5, 4) 
#         
#     def test_5_6(self):
#         self.execute(5, 6)
#     
#     def test_5_7(self):
#         self.execute(5, 7)   
#         
#     def test_5_8(self):
#         self.execute(5, 8) 
#     
#     def test_5_9(self):
#         self.execute(5, 9)
#         
#     def test_5_10(self):
#         self.execute(5, 10) 
#     
#     def test_5_11(self):
#         self.execute(5, 11)
            
            
    def loadImage(self, filename, factor=1):
        print(filename)
        imgT = cv2.imread(filename)
#         factor = 0.25
        shape = (round(factor * imgT.shape[1]), round(factor * imgT.shape[0]))
        imgMap = np.empty(shape, dtype='uint8')
        imgMap = cv2.resize(imgT, imgMap.shape)
        from scene.scene import Scene
        scene = Scene(imgMap)
        return scene
    
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
