# -*- coding: utf-8 -*-
from math import sqrt
from spottedObject import spottedObject
from calculations.labeling import LabelFactory

__author__ = 'tomek'
#/usr/bin/env python

'''

'''
import cv2
import numpy as np
import analyticGeometry as an
from skimage import morphology
from skimage import measure

class ContourDetector():
    
    
    def __init__(self,edge):
        self.edge = edge
        self.shape = self.edge.shape
    
    
    #
    # 
    # Contour detection
    #
    #
    
    def findContours(self):
        '''
        łaczy punkty w kontur
        '''
    
        contours = {0:[]}
        i = 0
        flag = True
    
        edge = self.edge.copy()
        nonzeros = np.nonzero(edge)
        pointX = nonzeros[1][0]
        pointY = nonzeros[0][0]
        contours[0].append((pointY,pointX))
    
        while flag == True:
            neibour = self.getNeibours(edge,pointX,pointY)
            #jeżeli znaleziono sąsiada tododaj go do konturu
            if neibour != False:
                edge[pointY,pointX] = 0
                pointX = neibour[0]
                pointY = neibour[1]
                contours[i].append((pointY,pointX))
    
            #jeżeli nie znaleziono sąsiada
            else:
                #poszukaj w otoczeniu
                edge[pointY,pointX] = 0
                near, dist = self.searchNearBy(edge,pointX,pointY)
                # near = np.asarray([])
                # dist = -1
    
                #jezeli znaleziono punkt w otoczeniu
                if near.size>0 :
                    pointX = near[0]
                    pointY = near[1]
    
                #jezeli nie znaleziono w otoczeniu żadnego punktu to znajdź pierwszy niezerowy
                else:
                    nonzeros = np.nonzero(edge)
                    if nonzeros[0].size > 0:
                        pointX = nonzeros[1][0]
                        pointY = nonzeros[0][0]
                        # cv2.circle(nimg, (pointX,pointY),10,(255,255,255),1)
                        # nimg[pointY,pointX] = (255,0,0)
    
                    # jeżeli nie ma żadnych niezerowych to zakończ algorytm szukania krawędzi
                    else:
                        flag = False
    
                # jeżeli punktu nie był w otoczeniu w odległości 5 to znaczy że to nowy kontur
                if dist != 5:
                    i+=1
                    # jeżeli to nie koniec algorytmu to zacznij nową krawędź
                    if flag != False:
                        contours[i]=[]
    
                # jeżeli to nie koniec algorytmu to dodaj punkt znaleziony dowolną meodą do bierzacego konturu
                if flag != False:
                    contours[i].append((pointY,pointX))
    
        # sklejenie konturow ktore powinny byc jednak razem
        neibours = []
        neibours = self.getContourNeibours(contours,contours[0][0])
        if len(neibours)>0:
            for n in neibours:
                if n[0] != 0:
                    contours[0].extend(contours[n[0]])
                    contours[n[0]] = []
    
#         print "ilosc"
#         print len(contours)
    
        return contours
    
    
    def getNeibours(self,edge,x,y):
    
        yMax,xMax = self.shape
    
        check = {}
    
        check['a'] = (x+1,y)
        check['c'] = (x-1,y)
        check['e'] = (x+1,y+1)
        check['g'] = (x-1,y-1)
    
        check['b'] = (x, y+1)
        check['d'] = (x,y-1)
        check['f'] = (x-1,y+1)
        check['h'] = (x+1,y-1)
    
    
        if x == 0:
            del check['c']
            del check['f']
            del check['g']
    
        elif x == xMax-1:
            del check['a']
            del check['e']
            del check['h']
    
        if y == 0:
            del check['d']
            if 'h' in check.keys():
                del check['h']
            if 'g' in check.keys():
                del check['g']
    
        elif y == yMax-1:
            del check['b']
            if 'e' in check.keys():
                del check['e']
            if 'f' in check.keys():
                del check['f']
    
        for value in check.itervalues():
            if edge[value[1],value[0]] == 255:
                return (value[0],value[1])
    
        return False
    
    
    def getContourNeibours(self,contours,(x,y)):
    
    
        check = {}
    
        check['a'] = (x+1,y)
        check['c'] = (x-1,y)
        check['e'] = (x+1,y+1)
        check['g'] = (x-1,y-1)
    
        check['b'] = (x, y+1)
        check['d'] = (x,y-1)
        check['f'] = (x-1,y+1)
        check['h'] = (x+1,y-1)
    
        neibours = []
        for k,v in contours.items():
            for point in check.itervalues():
                try:
                    index = v.index(point)
                except ValueError:
                    index = -1
                if index != -1:
                    neibours.append((k,index))
    
        return neibours
    
    
    def searchNearBy(self,edge,x,y):
#         print 'search'
        sp = edge.T.shape
    
        maskSizes = range(5,19,2)
        for i in maskSizes:
            mask = self.generateMask(i)
            points = np.asarray((x,y))+mask
    
            #filtowanie z poza granic
            xx = points[:,0]
            yy =points[:,1]
            XoverflowIndex = np.where(xx>sp[0]-1)
            YoverflowIndex = np.where(yy>sp[1]-1)
            wrongIndexes = np.union1d(XoverflowIndex[0],YoverflowIndex[0])
            points= np.delete(points,wrongIndexes,0)
    
            p = [edge[points[k][1],points[k][0]] for k in range(len(points))]
            non = np.nonzero(p)[0]
            if non.size>0:
                no = np.nonzero(p)[0][0]
                return points[no] , i
    
        return np.asarray([]),False
    
    
    def generateMask(self,maskSize):
    
        n = (maskSize-1)/2
        mask = []
        x=n
        y=n-1
        while x<n+1:
            while y<n:
                while x>-n:
                    while y>-n:
                        mask.append((x,y))
                        y-=1
                    mask.append((x,y))
                    x-=1
                mask.append((x,y))
                y+=1
            mask.append((x,y))
            x+=1
    
        return mask
    
    #
    # 
    # Corner detection
    #
    #   

    
    def findCornersOnContour(self,contour,size):
    
        if len(contour)>size:
            indexes = []
            dist = an.calcDistances(contour,size)
    
            for d in dist.iterkeys():
                segment1 = dist[d]
                MaxValue = max(np.asarray(segment1)[:,1])
                index = np.where(segment1 == MaxValue)[0]
                indexes.append(segment1[index[0]][0])
            return indexes
        else:
            return []
    
    
    def filterContours(self,contours,boundaries):
        '''
        pozbywa sie konturów z poza podanego obszaru
    
        contours - kontury - {0:[(a,b),(c,d)],1:[(e,f),(g,h),(j,k)]}
        boundaries - obszar graniczy - [[[1698  345]] \n\n [[1698  972]]]
    
        '''
        for key,c in contours.iteritems():
            if len(c)>0:
                isinside = cv2.pointPolygonTest(boundaries,(c[0][1],c[0][0]),0)
            else:
                isinside = 0
            if isinside != 1:
                contours[key] = []
            else:
                pass
        return contours
    
      
#     def eliminateSimilarCorners(self,corners,mainCnt,shape,border=35):
#         '''
#         eliminuje wierzchołki ktore prawdopodobnie sa blisko siebie
#         '''
#     
#         # płótno pod wierzchołki
#         nimg = np.zeros(shape,dtype='uint8')
#         nimg[:][:] = 0
#     
#         cornersInside = []
#         distances = {}
#     
#         #wybranie tylko tych wierzchołkow ktore leza wewnatrz obwiedni
#         for pt in corners:
#     
#             #sprawdź czy leży w konturze i w jakiej odległości
#             result = cv2.pointPolygonTest(mainCnt,pt,1)
#     
#             #jeżeli leżą wewnątrz
#             if result>=0:
#     
#                 #zpisz na listę
#                 cornersInside.append(pt)
#     
#                 #zapisz odległość
#                 distances[pt] = result
#     
#                 #zaznacz na płótnie
#                 nimg[pt[1],pt[0]] = 255
#     
#         # zabezpieczenie przed przelewaniem
#         img2 = cv2.copyMakeBorder(nimg,border,border,border,border,cv2.BORDER_CONSTANT,None,0)
#     
#         #słownik blixkich sobie wierzchołków
#         semi = {}
#     
#         # znalezienie bliskich sobie wierzchołków
#         for (x,y) in cornersInside:
#     
#             #wyodrębnij obszar
#             img3 = img2[y:y+border*2,x:x+2*border]
#     
#             #znajdź punkty na obszarze
#             non = np.nonzero(img3)
#     
#     
#             #jezeli jest więcej niż jeden
#             if len(non[0])>1:
#     
#                 semi[(x,y)] = []
#                 semi[(x,y)].append((x,y))
#     
#                 for k in range(len(non[0])):
#     
#                     #znajdź wektor między punktem a środkiem obszaru
#                     vecY = non[0][k]-border
#                     vecX = non[1][k]-border
#     
#                     #jezeli jest to wektor niezerowy to mamy punkt
#                     if (vecX != 0) or (vecY != 0):
#                         new = (x+vecX, y+vecY)
#                         semi[(x,y)].append(new)
#                         try:
#                             del cornersInside[cornersInside.index((x,y))]
#                         except ValueError:
#                             print 'error a'
#                             print (x,y)
#                             pass
#                         try:
#                             del cornersInside[cornersInside.index(new)]
#                         except ValueError:
#                             print 'error new'
#                             print (x,y)
#                             pass
#     
#         # wybranie wierzchołków bliższych konturowi zewnętrzenmu
#         for List in semi.itervalues():
#             dist = [distances[li] for li in List]
#             minIndex = dist.index(min(dist))
#             (x_,y_) = List[minIndex]
#     
#             # dodanie do listy globalnej wierzchołków
#             cornersInside.append((x_,y_))
#         return cornersInside
#     
#     
#     def findCorners(self,contours):
#     
#         # lista z podziełem na kontury
#         cornerCNT = {}
#     
#         #lista bez podziału na kontury
#         cornerList = []
#     
#         longestContour = []
#     
#         #dla każdego znalezionego konturu
#         for cindex in range(len(contours)):
#             cornerCNT[cindex] = []
#     
#             # szukanie najdłuższego - obiedni
#             cnt_len = len(contours[cindex])
#             if cnt_len>len(longestContour):
#                 longestContour = contours[cindex]
#     
#             indexes = self.findCornersOnContour(contours[cindex],16)
#     
#             # zaznacz wszsytkie znalezione wierzchołki
#             for Id in indexes:
#                 (y,x) = contours[cindex][Id]
#     
#                 # dodaj do globalnej listy ze wskazaniem konturu
#                 cornerCNT[cindex].append((x,y))
#     
#                 # dodaj do listy bez wskazania konturu
#                 cornerList.append((x,y))
#     
#             # usunięcie podobnych punktów na konturze
#             # cornerCNT[cindex] = eliminateSimilarCorners(cornerCNT[cindex],nimg,border=35)
#     
#         return cornerCNT,longestContour,cornerList
    
    
    def findObjects(self,contours):
        '''
         znajduje obiekty na podstawie konturów zmalezionych (łaczy poblisike kontury w jeden obiekt
        '''
    
        rectangles = []
        areas = []
        tmpbinary = np.zeros(self.shape,dtype='uint8')
        tmpbinary[:][:] = 0
    
        #zrób obramowania do każdego konturu
        for c in contours.itervalues():
            if len(c)>0:
                points = np.asarray([c])
                y,x,h,w = cv2.boundingRect(points)
    
                # powiększ obszar zainteresowania jeśli to możliwe i potrzebna
                # y = y-5;
                # x= x-5
                # w = w+10
                # h = h+10
    
                cont = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
                rectangle = np.asarray([cont])
                rectangles.append((rectangle))
    
        # narysuj białe obszary - w ten obszary stykające się lub nakładaące zleją sie w jeden obszar
        margin = int(round(tmpbinary.shape[1]*0.04,0))
        #margin = 70 #50 #10
    
        for r in rectangles:
            A = (r[0][0][0]-margin,r[0][0][1]-margin)
            B = (r[0][2][0]+margin,r[0][2][1]+margin)
            cv2.rectangle(tmpbinary,A,B,255,-1)
            # cv2.drawContours(tmpbinary,r,-1,255,-1)
    
            #znajdź kontury wśród białych prostokątów na czarnym tle
        cntTMP, h = cv2.findContours(tmpbinary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
        return cntTMP,margin
    
    
    def findMainObject(self,objectsCNT,img=0):
        '''
         znajduje obiekt najbardziej po srodku plaszczyzny (bo to nie beda krawedzie lustra)
        '''
        shape = self.shape
        #srodek obrazu
        yc0 = shape[0]/2
        xc0 = (shape[1]/4)*3
    
        min_index = -1
        min_cost = shape[0]*shape[1]
    
    
        for n,c in enumerate(objectsCNT):
            moments = cv2.moments(c)
            # policzenie srodkow ciezkosci figur
            yc = int(moments['m01']/moments['m00'])
            xc = int(moments['m10']/moments['m00'])
    
            #odległosc od srodka
            dx = xc0-xc
            dy = yc0-yc
            cost = sqrt(pow(dx,2)+pow(dy,2))
    
            if cost.real < min_cost:
                min_cost = cost.real
                min_index = n
        mainCNT = objectsCNT[min_index]
        
        mainObject = spottedObject(mainCNT)
    
        return mainCNT
    
    
def getLongest(contours):
    
    longestContour = []
    
    # szukanie najdłuższego - obiedni
    for cindex in range(len(contours)): 
        cnt_len = len(contours[cindex])
        if cnt_len>len(longestContour):
            longestContour = contours[cindex]
    
    return longestContour
 
 
def transfromEdgeMaskIntoEdges(edgeMask,emptyImage):
    '''
        szkieletyzacja maski konturow
    '''
    edges = morphology.skeletonize(edgeMask)
    edges = edges.astype('uint8')
    
    edges2 = morphology.skeletonize(edges)
    edges2 = edges2.astype('uint8')
    
    ei = emptyImage.copy()
    ei[:] = (0,0,0)
    ei[edges2 == 1] = (255,255,255)
    f = '../img/results/automated/9/objects2/debug/skeleton2.jpg' 
    print 'savaing to ' + f
    cv2.imwrite(f, ei)
        
        
    return edges2
     
     
class ObjectContourDetector():
    '''
        analizuje szkielet do postaci konturow
    '''
    
    
    def __init__(self,skeleton):
        self.skeleton = skeleton
    
    
    def fragmentation(self,skeleton):
        '''
            divide skeleton but the nodes
        '''
        points = np.nonzero(skeleton)
        points = np.transpose(points)
        nodes = []
        #znajdz wezly czyli punkty gdzie punkt ma wiecej niz 2 sasiadow
        for p in points:
            y = p[0]
            x = p[1]
            mask = skeleton[y-1:y+2,x-1:x+2]
            ones = np.nonzero(mask)
            neibours = ones[0].size
            if neibours>3:
                nodes.append((y,x))
            pass
        skeleton2 = skeleton.copy()
        for node in nodes:
            skeleton2[node] = 0
        
        #etykietyzacja krawedzi        
        lf = LabelFactory([])
        edgeLabelsMap = lf.getLabelsExternal(skeleton2, neighbors=8, background=0)
        edgeLabelsMap,edgeLabels,maxIT = lf.getBackgroundLabel(edgeLabelsMap,11,False)
        
        return skeleton2, edgeLabelsMap, edgeLabels, nodes