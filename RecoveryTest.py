# -*- coding: utf-8 -*-
'''
Created on Sep 27, 2014

@author: Tomasz
'''
import unittest
import numpy as np
import subprocess
import sys,os
import time
from IPython.utils.io import stdout

killed = False


class RecoveryTest(unittest.TestCase):
    
    LOG_DIR = 'results/log.'
    
    obiekty = {
               1: ["research/p2/DSC_0216.JPG","research/p2/DSC_0220.JPG","research/p2/DSC_0224.JPG","research/p2/DSC_0231.JPG"],
               2: ["research/p2/DSC_0217.JPG","research/p2/DSC_0221.JPG","research/p2/DSC_0225.JPG","research/p2/DSC_0230.JPG"],
               3: ["research/p2/DSC_0218.JPG","research/p2/DSC_0222.JPG","research/p2/DSC_0226.JPG","research/p2/poruszone.JPG"],
               4: ["research/p2/DSC_0219.JPG","research/p2/DSC_0223.JPG","research/p2/DSC_0227.JPG","research/p2/DSC_0228.JPG"]               
               }
    
    def runProcess(self,command):
        process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, env = { 'PYTHONIOENCODING': 'utf-8' })
    
        stdout = str()
        stderr = str()
    
        logFilename = self.LOG_DIR + '/lastTestRun.log'
        logFile = open(logFilename, 'w')
        logFile.close()
        global killed
        
        process.poll()
        
        while process.returncode is None and killed != True:
            # Wait for data to become available        
            stdoutPiece = process.stdout.readline()
            
            while stdoutPiece != '' and killed != True:
                print stdoutPiece,
                stdout += stdoutPiece
                logFile = open(logFilename, 'a')
                logFile.write(stdoutPiece)
                logFile.close()
                sys.stdout.flush()
                stdoutPiece = process.stdout.readline()
          
            process.poll()
        
        if killed == True:    
            killed = False
            process.kill()
            message = 'Test execution was killed!'
            logFile = open(logFilename, 'a')
            logFile.write(message)
            logFile.close()
            stderr = message
            print message
            
        else:
            stderr = process.stderr.readlines()
            stderr = ''.join(stderr)
            logFile = open(logFilename, 'a')
            logFile.write(stderr)
            logFile.close()
        
        returnCode = process.returncode    
        return (returnCode, stdout, stderr)


    def setUp(self):
        np.set_printoptions(precision=6,suppress=True)

    def tearDown(self):
        np.set_printoptions(precision=6)
        pass
    
    def rrun(self,obiekt,pozycja):
        filename = self.obiekty[obiekt][pozycja-1]
        command = 'python stereovision.py "localize" "'+filename+'"'
#         print command
        
        output = subprocess.check_output(command, shell=True)
        o2 = output[output.find('recovered2:')+13:]
        
        print 'obiekt:', obiekt, 'pozycja', pozycja
        print o2
        
        
#         self.runProcess(command)
        
#         os.system(command)
        
    def test_1_1(self):
        self.rrun(1,1)
        
    def test_1_2(self):
        self.rrun(1,2)
        
    def test_1_3(self):
        self.rrun(1,3)
        
    def test_2_1(self):
        self.rrun(2,1)
     
    def test_2_2(self):
        self.rrun(2,2)
         
    def test_2_3(self):
        self.rrun(2,3)
         
    def test_3_1(self):
        self.rrun(3,1)
     
    def test_3_2(self):
        self.rrun(3,2)
         
    def test_3_3(self):
        self.rrun(3,3)
         
    def test_4_1(self):
        self.rrun(4,1)
     
    def test_4_2(self):
        self.rrun(4,2)
         
    def test_4_3(self):
        self.rrun(4,3)
        
    def test_1_4(self):
        self.rrun(1,4)
        
    def test_2_4(self):
        self.rrun(2,4)
        
    def test_3_4(self):
        self.rrun(3,4)
    
    def test_4_4(self):
        self.rrun(4,4)
       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    