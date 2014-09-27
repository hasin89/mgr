# -*- coding: utf-8 -*-
from ctypes.wintypes import LONG

__author__ = 'tomek'
#/usr/bin/env python

'''

'''

from collections import Counter
from math import sqrt
from math import cos
import cv2
import numpy as np

class LineDetector():
    
    
    def __init__(self,contour,shape):
        """
            contour - kiedys byl najdluzszym
            shape - obszar na ktorym znajduje sie kontur
        """
        self.contour = contour
        self.shape = self.shape
        
        self.treshhold = 25 #125
        self.theta = 0.025 
        self.rho = 1
    
    
    def findLines(self):
        '''
        zwraca linie Hougha na podstawie podanego konturu
        
        longestContour - podany kontur
        shape - kształt płótna
        
        return lines - linie
        '''
        longestContour = self.contour
        
        tmpbinary = np.zeros(self.shape,dtype='uint8')
        tmpbinary[:][:] = 0
        
        for c in longestContour:
            tmpbinary[c] = 1
        
        rho = self.rho
        # theta = 0.025
        theta = self.theta
        #znaldź linie hougha
        lines2 = cv2.HoughLines(tmpbinary,rho,theta,self.threshold)
        if lines2 != None:
            lines = lines2[0]
            lines = self.eliminateSimilarLines(lines)
        else:
            lines = False
        return lines
    
    def eliminateSimilarLines(self,linesNP):
        """
            eliminuje linie zbyt podobne do siebie
        """
        lines = linesNP.tolist()
        similar = {}
        #szukanie lini o małym wzajemnym kacie nachylenia (czyli o podobnym wgl osi OY)
        for L in lines:
            similar[L[1]] = []
            for line in lines:
                if (line[0] != L[0]) and (line[1] != L[1]):
                    x = abs(L[1]-line[1])
                    value = cos(x)
                    #TODO: - jeśli znajdzie się para prostych równoległych,
                    # ale odległych o znaczną odległość (równoległe boki) to trzeba temu bedzie zaradzić
                    if value>0.9:
                        #JEŚLI SĄ RÓWNOLEGŁE ALE NIE BLISKO SIEBIE TO NIE LICZ ICH
                        if value>0.999 and abs(line[0]-L[0])>20 :
                            pass
                        else:
                            # pass
                            similar[L[1]].append(line[1])
    
        # wybieranie lini która jest najbliższa średniej z lini o podobnym kącie nachylenia
        flag = True
        while (flag):
            max = 0
            kmax = 0
    
            for k,v in similar.iteritems():
                ll = len(v)
                if ll > max:
                    max = ll
                    kmax =  k
    
            if kmax!=0:
                for k in similar[kmax]:
                    if k in similar.keys():
                        del similar[k]
                similar[kmax].append(kmax)
                mean = np.mean(similar[kmax])
                min = 4
                kmin = -1
                for i in similar[kmax]:
                    if abs(i-mean)<min:
                        min = abs(i-mean)
                        kmin = i
    
                del similar[kmax]
                similar[kmin] = []
            else:
                flag = False
    
        finalLines = {}
    
        for l in lines:
            if l[1] in similar.keys():
                finalLines[l[0]] = l
    
        # sprawdzenie czy nie znalazły się jakieś równoległe do siebie linie i jeśli tak to wybranie średniej z nich
        ff = [f[1] for f in finalLines.values()]
        c = Counter(ff)
        for k,v in c.iteritems():
            if v == 1:
                pass
            else:
                print k
                suma = 0
                indexes = []
                for key,value in finalLines.iteritems():
                    if value[1] == k:
                        suma += value[0]
                        indexes.append(key)
                if len(indexes)>0:
                    mean = int(suma/v)
                    for i in indexes:
                        del finalLines[i]
                    finalLines[mean] = [mean,k]
    
        lines = finalLines.values()
    
        return lines
    
