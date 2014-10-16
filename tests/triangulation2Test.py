'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
from calculations import triangulation,DecompPMat


class triangulationTest(unittest.TestCase):


    def setUp(self):
        np.set_printoptions(precision=4)
        
        self.Points = []
        self.Points2 = []
        
        x1 = [1908,2128,1595,1820,1910,1605]
        y1 = [2040,1842,1900,1737,2365,2225]
        
        x2 = [737,1007,885,1148,777,915]
        y2 = [1419,1336,1514,1434,1677,1779]
        
        X = [0,2,0,2,0,0]
        Y = [0,0,2,2,0,2]
        Z = [0,0,0,0,2,2]
        
#         wzorzec bezp:
        x1 = [3066,2731,3089,2977,2745,2859]
        y1 = [1532,1595,1349,1493,1405,1262]
        
#         obraz
        x2 = [ 2912,2650,2924,2804,2658,2772]
        y2 = [ 794, 772, 637, 554, 611, 697]

        X = [0 ,27 ,0 ,11,27,17]
        Y = [13,19 ,13,29,19,2 ]
        Z = [0 ,0  ,20,20,20,20]
        
        
        for i in range(len(x1)):
            element = ( (X[i],Y[i],Z[i]),(x1[i],y1[i]) )
            self.Points.append(element)

        for i in range(len(x2)):
            element = ( (X[i],Y[i],Z[i]),(x2[i],y2[i]) )
            self.Points2.append(element)


    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    
    def testCameraMatrixShape(self):
        camera_matrix = triangulation.calculate_camera_matrix(self.Points)
        self.assert_(camera_matrix.shape == (3,4))
        
        
    def testCameraMatrixValues(self):
        actual = triangulation.calculate_camera_matrix(self.Points)
        expected = np.matrix( [[  1.96411305e+02,  -1.06890790e+02,   4.50828120e+01,   1.90996543e+03],
                               [ -1.89813300e+01,  -3.54226759e+00,   2.22496437e+02,   2.03497915e+03],
                               [  4.12986805e-02,   3.23351992e-02,   2.38527090e-02,   1.00000000e+01]])
        self.assert_(str(actual) == str(expected),"example camera matrixes are not equal:\n"+str(actual)+"\n Expected:\n"+str(expected))


    def testProcessPointsValues(self):
        
        actual = triangulation._process_points(self.Points)
        expected = np.matrix([[    0,     0,     0,     1,     0,     0,     0,     0,      0,     0,     0],
                              [    2,     0,     0,     1,     0,     0,     0,     0,  -4256,     0,     0],
                              [    0,     2,     0,     1,     0,     0,     0,     0,     0,  -3190,     0],
                              [    2,     2,     0,     1,     0,     0,     0,     0, -3640,  -3640,     0],
                              [    0,     0,     2,     1,     0,     0,     0,     0,     0,      0, -3820],
                              [    0,     2,     2,     1,     0,     0,     0,     0,     0,  -3210, -3210],
                              [    0,     0,     0,     0,     0,     0,     0,     1,     0,      0,     0],
                              [    0,     0,     0,     0,     2,     0,     0,     1, -3684,      0,     0],
                              [    0,     0,     0,     0,     0,     2,     0,     1,     0,  -3800,     0],
                              [    0,     0,     0,     0,     2,     2,     0,     1, -3474,  -3474,     0],
                              [    0,     0,     0,     0,     0,     0,     2,     1,     0,      0, -4730],
                              [    0,     0,     0,     0,     0,     2,     2,     1,     0,   -4450,-4450]])
        self.assert_((actual == expected).all(),"example matrixes are not equal")

    def testDecompositionK(self):
        K_ = np.matrix([[  3.56282934e+03,   0.00000000e+00,   1.72601538e+03],
                        [  0.00000000e+00,   3.64137753e+03,   1.32788223e+03],
                        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
        P = triangulation.calculate_camera_matrix(self.Points)
        K,R,t = DecompPMat.decomposeProjectionMatrix(P)
        self.assert_(str(K)==str(K_),"K is not as expected")
        pass 
    
    
    def testDecompositionR(self):
        R_ = np.matrix([[ 0.6058, -0.7951,  0.0289 ],
                        [ -0.3453,-0.2301,  0.9098 ],
                        [ 0.7167,  0.5612,  0.4140]])
        P = triangulation.calculate_camera_matrix(self.Points)
        K,R,t = DecompPMat.decomposeProjectionMatrix(P)
        self.assert_(str(R)==str(R_),"R is not as expected")
        pass
    
    
    def testDecompositionT(self):
        t_ = np.matrix([[ -74.77273186],
                        [ -53.58875054],
                        [ 173.54997624]])
        P = triangulation.calculate_camera_matrix(self.Points)
        K,R,t = DecompPMat.decomposeProjectionMatrix(P)
        self.assert_(str(t)==str(t_),"t is not as expected")
        pass
    
    
    def testRecovery_3dX(self):
        P1 = triangulation.calculate_camera_matrix(self.Points)
        P2 = triangulation.calculate_camera_matrix(self.Points2)
        
        p1 = (1855,2625)
        p2 = (517,1700)

        X,Y,Z = triangulation.recovery_3d(P1,P2,p1,p2)
        X_expected = -1.537
        self.assert_(str(X)[:6]==str(X_expected),"X is not as expected:"+str(X)+" instead:"+str(X_expected))
        pass
    
    
    def testRecovery_3dY(self):
        P1 = triangulation.calculate_camera_matrix(self.Points)
        P2 = triangulation.calculate_camera_matrix(self.Points2)
        
        p1 = (1855,2625)
        p2 = (517,1700)
        
        #driuga opcja        
        p1 = (1540,1563)
        p2 = (1764,810)
        
        X,Y,Z = triangulation.recovery_3d(P1,P2,p1,p2)
        Y_expected = -0.783 
        self.assert_(str(Y)[:6]==str(Y_expected),"Y is not as expected:"+str(Y)+" instead:"+str(Y_expected))
        pass
    
    
    def testRecovery_3dZ(self):
        P1 = triangulation.calculate_camera_matrix(self.Points)
        P2 = triangulation.calculate_camera_matrix(self.Points2)
        
        p1 = (1855,2625)
        p2 = (517,1700)
        
        X,Y,Z = triangulation.recovery_3d(P1,P2,p1,p2)
        Z_expected = 1.997 
        self.assert_(str(Z)[:5]==str(Z_expected),"X is not as expected:"+str(Z)+" instead:"+str(Z_expected))
        pass
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()