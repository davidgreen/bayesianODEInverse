#DKE Green
#2019

import math
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

import sys
sys.path.append('../BayesTF')
import bayesTF



def printData(headerText,data,numTimes):
    print("=" * 80)
    print(headerText)
    print("=" * 80)

    print("t traceX traceY traceZ")
    for i in range(0,numTimes):
        print("%e %e %e %e" % (data['times'][i],data['data'][i,0],data['data'][i,1],data['data'][i,2]))


def main():

    np.random.seed(108976)
    tf.random.set_random_seed(108976)


    ## GENERATE TRAINING DATA TRACES
    traceGen = bayesTF.SprottB()

    timeInterval = [0,100]
    numTimes = 1000

    rtol = 1e-16

    initX = traceGen.generateInitialConditions(x0=1.0,y0=1.0,z0=1.0)
    data = traceGen.generateDataStream(initVals=initX,numTimes=numTimes,timeInterval=timeInterval)

    #Test data
    initXTest = traceGen.generateInitialConditions(x0=-1.0,y0=-1.0,z0=-1.0)
    dataTest = traceGen.generateDataStream(initVals=initXTest,numTimes=numTimes,timeInterval=timeInterval)

    initXExample = traceGen.generateInitialConditions(x0=0.1,y0=1.0,z0=-0.1)
    dataExample = traceGen.generateDataStream(initVals=initXExample,numTimes=numTimes,timeInterval=timeInterval)


    printData(headerText="Output train trace data",data=data,numTimes=numTimes)
    printData(headerText="Output test trace data",data=dataTest,numTimes=numTimes)
    printData(headerText="Output example trace data",data=dataExample,numTimes=numTimes)


if __name__ == "__main__":
    main()
