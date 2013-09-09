from unittest.test.test_result import __init__

__author__ = 'tomek'

import cv2
# import numpy as np

class Segment(object):
    def __init__(self):
        self.length = 0
        self.neibours = {}
        self.points = {}

class Polyline(object):
    def __init__(self):
        self.segments = {}
        self.begining = (0,0)
        self.ending = (0,0)
