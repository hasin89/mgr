# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
#  DecompPMat - decompose the camera projection matrix P into intrinsic
#               matrix K, rotation matrix R, and translation vector t.
# 
#  Usage:
#            [K,R,t] = DecompPMat(P)
# 
#  Input:
#            P : 3x4 camera projection matrix
# 
#  Output:
#            K : 3x3 intrinsic matrix
#            R : 3x3 rotation matrix
#            t : 3x1 translation vector
# 
#  cf.:
#            P = K*[R t]
# 
#  This code follows the algorithm given by
#  [1] "Summary Note 2006-SN-003-EN_Camera Calibration, ISRC, SKKU"
#      available at http://www.daesik80.com
#  [2] E. Trucco and A. Verri, "Introductory Techniques for 3-D Computer Vision,"
#      Prentice Hall, pp.134-135, 1998.
# 
#  Kim, Daesik
#  Intelligent Systems Research Center
#  Sungkyunkwan Univ. (SKKU), South Korea
#  E-mail  : daesik80@skku.edu
#  Homepage: http://www.daesik80.com
# 
#  June 2008  - Original version.
import numpy as np
from numpy.lib.scimath import sqrt
from numpy.linalg.linalg import svd



def decomposeProjectionMatrix(P):
    """
        P - projection matrix
        
        output
            K : 3x3 intrinsic matrix
            R : 3x3 rotation matrix
            t : 3x1 translation vector        
    """
    
    K = np.matrix([[0.0, 0, 0],[0, 0, 0],[0, 0, 1]])
    R = np.matrix([[0.0, 0, 0],[0, 0, 0],[0, 0, 0]])
    t = np.matrix([[0.0], [0], [0]])
    
    # Normalized projection
    
    x = pow(P[2,0],2) + pow(P[2,1],2) + pow(P[2,2],2)
    scale = sqrt(x)
    P = P/scale
    
    # Principal point
    K[0,2] = P[0,:3]*P[2,:3].transpose()
    K[1,2] = P[1,:3]*P[2,:3].transpose() 
    
    # Focal length
    K[0,0] = sqrt( P[0,:3]*P[0,:3].transpose() - pow(K[0,2],2) )
    K[1,1] = sqrt( P[1,:3]*P[1,:3].transpose() - pow(K[1,2],2) )
    
    # Translation vector
    t[2,0] = P[2,3]
    t[0,0] = ( P[0,3]-K[0,2]*t[2,0] ) / K[0,0]
    t[1,0] = ( P[1,3]-K[1,2]*t[2,0] ) / K[1,1]
    
    # Rotation matrix
    R[2,0] = P[2,0]
    R[2,1] = P[2,1]
    R[2,2] = P[2,2]
    
    R[0,:] = ( P[0,:3] - K[0,2] * P[2,:3] ) / K[0,0] 
    R[1,:] = ( P[1,:3] - K[1,2] * P[2,:3] ) / K[1,1]
    
    # Orthogonality Enforcement
    [U,D,V] = svd(R)
    D = np.eye(3)
    R = U*D*V
    
    #Tz Sign fixing
    if t[2,0] < 0:
        t = -t
        R = -R
        
    return K,R,t