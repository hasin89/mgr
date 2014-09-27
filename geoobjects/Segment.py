'''
Created on Sep 4, 2014

@author: Tomasz
'''
import scene.analyticGeometry as analyticGeometry

class Segment(object):
    def __init__(self):
        self.length = 0
        self.neibourLines = []
        self.points = []
        self.line = (0,0,0)

    def calcLength(self):
        if len(self.points) == 2:
            p1 = self.points[0]
            p2 = self.points[1]
            self.length = analyticGeometry.calcLength(p1,p2)
        else:
            self.length = 0

    def setPoints(self,p1,p2):
        self.points = [p1,p2]
        self.calcLength()
        self.getLine()

    def getLine(self):
        if len(self.points) == 2:
            p1 = self.points[0]
            p2 = self.points[1]
            self.line = analyticGeometry.getLine(p1,p2,0)
        else:
            self.line = (0,0,0)