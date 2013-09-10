from unittest.test.test_result import __init__

__author__ = 'tomek'

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


class Polyline(object):
    def __init__(self):
        self.segments = {}
        self.points = []
        self.begining = (0,0)
        self.ending = (0,0)

class Vertex(object):
    def __init__(self):
        self.lines = []
        self.point = (0,0)


