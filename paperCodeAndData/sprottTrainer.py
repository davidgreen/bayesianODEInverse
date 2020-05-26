#DKE Green
#2019

import numpy as np


import sys
sys.path.append('../BayesTF')
import bayesTF

from tqdm import tqdm

import sympy
from sympy.matrices import *
from sympy import pprint


from fixedNoiseANNForwardMaskWeightsSprott import FixedNoiseANNForwardMaskWeightsSprott
from learnSprottANNDrvApproxDropOutWeights import LearnSprottANNDrvApproxDropOutWeights

class SprottTrainer(object):

    def __init__(self):

        self.traceGen = bayesTF.SprottB()
        self.timeInterval = [0,100]


        self.initX = self.traceGen.generateInitialConditions(x0=1.0,y0=1.0,z0=1.0)
        self.initXTest = self.traceGen.generateInitialConditions(x0=-1.0,y0=-1.0,z0=-1.0)


    def generateOutput(self,numTimes,keepProb,numTraces):
        data,dataTest,approxDrvs,inputUs = self.__generateTrainAndTestData(numTimes)

        h = (self.timeInterval[1]-self.timeInterval[0])/float(numTimes)
        nn = self.__setupANN(keepProb=keepProb)

        self.__runTraining(nn=nn,numTrainingTimes=int(numTimes/10),h=h,inputUs=inputUs,approxDrvs=approxDrvs)

        return data,dataTest,self.__generateTraces(nn=nn,numTraces=numTraces,h=h,keepProb=keepProb,numTimes=numTimes)





    def __generateTrainAndTestData(self,numTimes):
        data = self.traceGen.generateDataStream(initVals=self.initX,numTimes=numTimes,timeInterval=self.timeInterval)#,rtol=rtol)#,atol=atol)
        dataTest = self.traceGen.generateDataStream(initVals=self.initXTest,numTimes=numTimes,timeInterval=self.timeInterval)#,rtol=rtol)#,atol=atol)
        approxDrvs,inputUs = self.__extractApproxDrvs(data=data['data'],times=data['times'])
        return data,dataTest,approxDrvs,inputUs


    def __extractApproxDrvs(self,data,times):

        diffs = np.diff(data,axis=0)
        divDTs = (times[1:] - times[:-1])

        approxDrvs = np.zeros(diffs.shape)
        for i in range(0,approxDrvs.shape[1]):
            approxDrvs[:,i] = diffs[:,i] / divDTs

        inputUs = np.zeros((data.shape[0]-1,data.shape[1]))

        for i in range(0,inputUs.shape[0]):
            inputUs[i,:] = data[i-1,:]

        return approxDrvs,inputUs



    def __setupANN(self,keepProb):
        nn = LearnSprottANNDrvApproxDropOutWeights(name="nn",inputDim=3,outputDim=3,keepProb=keepProb)
        nn.initSession()
        return nn


    def __runTraining(self,nn,numTrainingTimes,h,inputUs,approxDrvs):
        baseTrainingUs = inputUs[:numTrainingTimes,:]
        baseTrainingDrvs = approxDrvs[:numTrainingTimes,:]
        for i in range(0,1000):
            ys = baseTrainingDrvs + np.random.normal(0,h**1.0,baseTrainingDrvs.shape)
            nn.trainValues(numRepeats=10,learningRate=0.01,
                xs=baseTrainingUs,ys=ys)
            nn.trainValues(numRepeats=100,learningRate=0.001,
                xs=baseTrainingUs,ys=ys)




    def __generateTraces(self,nn,numTraces,h,keepProb,numTimes):
        ##TO SPEED THIS UP: need to get the dropout mask...
        #instead of just always applying droput every step.
        #https://stackoverflow.com/questions/37463863/how-to-get-the-dropout-mask-in-tensorflow
        print("Starting trace gen")
        traceGen2 = FixedNoiseANNForwardMaskWeightsSprott(nn,h=h,keepProb=keepProb)

        allOutputDataArray = np.zeros((numTraces,numTimes,3))

        for i in range(0,numTraces):
            keepRunning = True

            while(keepRunning):
                #find one that doesnt blow up
                print(i)
                traceGen2.setMask();

                data2 = traceGen2.generateDataStream(initVals=self.initXTest,numTimes=numTimes,timeInterval=self.timeInterval)

                dataLen = len(data2['data'][:,0])

                if(dataLen == numTimes):
                    keepRunning = False;
                    # numValsPerTrace[i] = dataLen
                    allOutputDataArray[i,:dataLen,:] = data2['data'][:,:]

        return allOutputDataArray
