'''
Created on Feb 13, 2015

@author: Tomasz
'''
import distortion
import numpy as np

def scalar_product(A,B):
    ashape = A.shape
    bshape = B.shape
    
    
    if ashape[0]<ashape[1]:
        A = A.T
        result = np.zeros((1,ashape[1]))
        length = ashape[1]
    else:
        result = np.zeros((1,ashape[0]))
        length = ashape[0]
    for i in range(length):
        product = np.dot(A[i],B[:,i])
        result[0,i] = product
    return result

def triangulate(leftPoints,rightPoints,mtx,dist,R,t):
    #normalization
    l_n = distortion.normalize(leftPoints, mtx, dist)
    r_n = distortion.normalize(rightPoints, mtx, dist)
    
    #number of points
    N = max(l_n.shape)
    
    #to homogeneous coordinates
    l_nh = np.array([l_n[0],l_n[1],np.tile(1,N)])
    r_nh = np.array([r_n[0],r_n[1],np.tile(1,N)])
        
    u = np.dot(R,l_nh)
    
    n_l2 = scalar_product(l_nh, l_nh)
    n_r2 = scalar_product(r_nh, r_nh)
    
    t = t.reshape((3,1))
    T_vec = np.tile(t,N)
    
    DD = n_l2 * n_r2 - scalar_product(u, r_nh)**2
    
    dot_uT = scalar_product(u, T_vec)
    dot_xrT = scalar_product(r_nh, T_vec)
    dot_xru = scalar_product(r_nh, u)

if __name__ == '__main__':
    pass