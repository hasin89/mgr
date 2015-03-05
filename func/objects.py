from unittest.test.test_result import __init__

__author__ = 'tomek'
raise
import cv2
import analise as an
# import numpy as np

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
            self.length = an.calcLength(p1,p2)
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
            self.line = an.getLine(p1,p2,0)
        else:
            self.line = (0,0,0)


class Polyline(object):
    def __init__(self):
        self.segments = {}
        self.points = []
        self.begining = (0,0)
        self.ending = (0,0)

class Vertex(object):
    def __init__(self,p):
        self.lines = []
        self.point = p
        self.neibours = {}

    def __str__(self):
        return "v( %d , %d )" % (self.point[0],self.point[1])

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if (other.__class__.__name__ == 'Vertex') & (other.point == self.point):
            return True
        else:
            return False



class Face(object):
    def __init__(self):
        self.vertices = []

