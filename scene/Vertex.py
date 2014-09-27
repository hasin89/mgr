# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
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