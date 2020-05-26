#DKE Green
#2019

import math
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../BayesTF')
import bayesTF


class FixedNoiseANNForwardMaskWeightsSprott(bayesTF.ScipyODEIntegrator):
    def __init__(self,trainedNN,h,keepProb):

        self.keepProb = keepProb
        self.h = h

        weights = trainedNN.getAllWeights()


        self.denseW0 = weights['dense/kernel:0']
        self.denseB0 = weights['dense/bias:0'].reshape((-1,1))
        self.denseW1 = weights['dense_1/kernel:0']
        self.denseB1 = weights['dense_1/bias:0'].reshape((-1,1))

        self.w = self.denseB0.shape[0]


    def setMask(self):

        self.mask = np.random.choice(2, self.w, p=[1.0-self.keepProb, self.keepProb])
        self.mask = self.mask.reshape((-1,1))


        self.noise = np.random.normal(loc=0.0,scale=self.h**4,size=3)


    def odeDerivative(self,t,y):

        yrs = y.reshape((-1,1))

        #For speed during some earlier testing, weights were manually pulled out from tensorflow post training.
        l1 = self.denseW0.transpose().dot(yrs) + self.denseB0
        l1 = self.mask*l1 #put in mask here
        l1P = l1*l1
        l2 = self.denseW1.transpose().dot(l1P) + self.denseB1


        self.noise = np.random.normal(loc=0.0,scale=self.h**4,size=3)


        retDerivatives = l2.reshape(3) + self.noise
        return retDerivatives
