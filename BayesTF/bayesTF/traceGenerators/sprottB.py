#DKE Green
#2019

import math
import matplotlib.pyplot as plt
import numpy as np

from .scipyODEIntegrator import ScipyODEIntegrator

class SprottB(ScipyODEIntegrator):
    def __init__(self):

        #dx/dt = yz, dy/dt = x-y, dz/dt = 1-xy
        self.odeCalculator = ScipyODEIntegrator(odeDerivative=self.odeDerivative)


    def __assembleEquations(self,x,t):
        #dx/dt = yz, dy/dt = x-y, dz/dt = 1-xy
        dxdt = x[1]*x[2]
        dydt = x[0]-x[1]
        dzdt = 1.-x[0]*x[1]
        return np.array([dxdt,dydt,dzdt])


    def odeDerivative(self,t,y):
        return self.__assembleEquations(x=y,t=t)


    def generateInitialConditions(self,x0,y0,z0):
        return np.array([x0,y0,z0])
