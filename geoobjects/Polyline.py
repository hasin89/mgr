# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
class Polyline(object):
    def __init__(self):
        self.segments = {}
        self.points = []
        self.begining = (0,0)
        self.ending = (0,0)