'''
Created on Feb 13, 2015

@author: Tomasz
'''
import numpy as np

def undistort(points,dist):
    '''
    points - xsy w jednym wierszu, yki w drugim
    '''
    points_length = max(points.shape)
    points = points.reshape((2,points_length))
    
    length = max(dist.shape)
    
    if length == 1:
        # comp_distortion.m from calibration toolbox
        k = dist.ravel()[0]
        x = points[0]
        y = points[1]
        radius_2 = x**2 + y**2
        b = np.dot(k,radius_2).reshape(1,points_length)
        radial_dist = 1+ np.dot(np.ones((2,1)),b)
        radius_2_comp = ( x**2 + y**2 ) / radial_dist[0] 
        radial_dist = 1+ np.dot( np.ones((2,1)) , np.dot(k,radius_2_comp).reshape((1,points_length)) )
        x_comp = np.divide(points,radial_dist)
        
#         return x_comp
    
    else:
        #comp_distortion_oulu.m from calibration toolbox
        k1 = dist[0]
        k2 = dist[1]
        k3 = dist[2]
        p1 = dist[3]
        p2 = dist[4]
        
        x_comp = points
        
        for i in range(1,21):
            x1 = x_comp[0]
            y1 = x_comp[1]
        
            r_2 = sum(x_comp**2)
            k_radial =  1 + k1 * r_2 + k2 * r_2**2 + k3 * r_2**3
            delta_x1 = 2*p1*x1*y1 + p2*(r_2 + 2*x1**2)
            delta_x2 = p1 * (r_2 + 2*y1**2)+2*p2*x1*y1
            delta_x = np.array([delta_x1,delta_x2])
            x_comp = (points - delta_x) / ( np.dot( np.ones((2,1)) , k_radial.reshape((1,points_length))))
        return x_comp
    
def normalize(points,mtx,dist):
    a = (points[0]-mtx[0,2]) / mtx[0,0]
    b = (points[1]-mtx[1,2]) / mtx[1,1]
    x_d = np.array([[a],[b]])
    
    x_n = undistort(x_d, dist)
    print x_n
    

if __name__ == '__main__':
    pass
