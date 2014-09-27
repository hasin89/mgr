# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
import numpy as np

x1 = [1908,2128,1595,1820,1910,1605]
y1 = [2040,1842,1900,1737,2365,2225]

x2 = [737,1007,885,1148,777,915]
y2 = [1419,1336,1514,1434,1677,1779]

X = [0,2,0,2,0,0]
Y = [0,0,2,2,0,2]
Z = [0,0,0,0,2,2]

Points = []

for i in range(len(x1)):
    element = ( (X[i],Y[i],Z[i]),(x1[i],y1[i]) )
    Points.append(element)
    
    
def calculate_camera_matrix(Points):
    """
        Points są postaci: [((X1,Y1,Z1),(imgX1,imgY1)), ]
    """
    c = process_points(Points)
    inv_c = np.linalg.pinv(c)
    
    x_series = []
    y_series = []
     
    for (X,Y,Z),(imgX,imgY) in Points:
        x_series.append(imgX)
        y_series.append(imgY)
    
    b = x_series
    b.extend(y_series)
    bb = np.matrix(b)
    bb_T = bb.transpose()
    
    CM = inv_c * bb_T
    
    camera_projection_matrix= np.matrix([
                                          [CM[0,0], CM[1,0],  CM[2,0], CM[3,0]],
                                          [CM[4,0], CM[5,0],  CM[6,0], CM[7,0]],
                                          [CM[8,0], CM[9,0], CM[10,0],    10  ]
                                          ])
    return camera_projection_matrix


def process_points(Points): 
    a = []
    
    # %      A = [
    # %      X(1), Y(1), Z(1) , 1, 0, 0, 0, 0, -xI(1)*X(1), -xI(1)*Y(1), -xI(1)*Z(1);
    # %      X(2), Y(2), Z(2) , 1, 0, 0, 0, 0, -xI(2)*X(2), -xI(2)*Y(2), -xI(2)*Z(2);
    # %      X(3), Y(3), Z(3) , 1, 0, 0, 0, 0, -xI(3)*X(3), -xI(3)*Y(3), -xI(3)*Z(3);
    # %      X(4), Y(4), Z(4) , 1, 0, 0, 0, 0, -xI(4)*X(4), -xI(4)*Y(4), -xI(4)*Z(4);
    # %      X(5), Y(5), Z(5) , 1, 0, 0, 0, 0, -xI(5)*X(5), -xI(5)*Y(5), -xI(5)*Z(5);
    # %      X(6), Y(6), Z(6) , 1, 0, 0, 0, 0, -xI(6)*X(6), -xI(6)*Y(6), -xI(6)*Z(6);
    # % 
    # %      0, 0, 0, 0, X(1), Y(1), Z(1) , 1, -yI(1)*X(1), -yI(1)*Y(1), -yI(1)*Z(1);
    # %      0, 0, 0, 0, X(2), Y(2), Z(2) , 1, -yI(2)*X(2), -yI(2)*Y(2), -yI(2)*Z(2);
    # %      0, 0, 0, 0, X(3), Y(3), Z(3) , 1, -yI(3)*X(3), -yI(1)*Y(3), -yI(3)*Z(3);
    # %      0, 0, 0, 0, X(4), Y(4), Z(4) , 1, -yI(4)*X(4), -yI(1)*Y(4), -yI(4)*Z(4);
    # %      0, 0, 0, 0, X(5), Y(5), Z(5) , 1, -yI(5)*X(5), -yI(1)*Y(5), -yI(5)*Z(5);
    # %      0, 0, 0, 0, X(6), Y(6), Z(6) , 1, -yI(6)*X(6), -yI(1)*Y(6), -yI(6)*Z(6)
    # %     ]
    
    for (X,Y,Z),(imgX,imgY) in Points:
        a.append([X,Y,Z,1,0,0,0,0,-imgX*X,-imgX*Y,-imgX*Z])
        
    for (X,Y,Z),(imgX,imgY) in Points:
        a.append([0,0,0,0,X,Y,Z,1,-imgY*X,-imgY*Y,-imgY*Z])    
    return np.matrix(a)


def recovery_3d():
    
    pass

calculate_camera_matrix(Points)