'''
Created on Sep 4, 2014

@author: Tomasz
'''

class Point(object):
    '''
    classdocs
    '''
    POINT_TYPE = {
             'VERTEX':0,
             '':1
             }


    def __init__(self,x,y):
        '''
        Constructor
        '''
        self.x = x
        self.y = y
        self.type = None
        
    def __str__(self):
        return "p( %d , %d )" % (self.x,self.y)

    def __repr__(self):
        return self.__str__()