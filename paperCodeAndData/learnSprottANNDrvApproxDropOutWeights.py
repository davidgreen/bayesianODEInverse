#DKE Green
#2019

import numpy as np
import tensorflow as tf


import sys
sys.path.append('../BayesTF')
import bayesTF

from tqdm import tqdm

import sympy
from sympy.matrices import *
from sympy import pprint


class LearnSprottANNDrvApproxDropOutWeights(bayesTF.ProbabilisticNNManagerBase):

    def __init__(self,name,inputDim,outputDim,keepProb):

        self.keepProb = keepProb
        self.numType = tf.float32

        bayesTF.ProbabilisticNNManagerBase.__init__(self,name,inputDim,outputDim)


    def _setupNetwork(self):
        self.inputX = tf.placeholder(self.numType, shape=[None,self.inputDim]) #u(t-1)
        self.inputDT = tf.placeholder(self.numType, shape=[None, 1])

        polyOrder = 2
        width = 10 #100
        nn = tf.layers.dense(self.inputX,width)


        self.nnTempFirst = nn

        l1 = nn
        nn = tf.nn.dropout(l1,self.keepProb)
        nn = l1*l1

        self.nnTemp = nn


        nn = tf.layers.dense(nn,self.outputDim)

        self.output = nn




    def _setupTraining(self):



        self.outputTrue = tf.placeholder(self.numType, shape=[None, self.outputDim])
        self.lossTerm = tf.reduce_mean(tf.square(self.output-self.outputTrue))

        self.lr = tf.get_variable(self.name + "_lr",initializer=tf.constant(0.01,self.numType),trainable=False)


        self.loss = self.lossTerm

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train = self.optimizer.minimize(self.loss)


    def predict(self,xs):

        feed_input = {
            self.inputX: xs
        }

        outputVars = [self.output]
        outputValues = self.sess.run(outputVars,feed_dict=feed_input)

        return outputValues[0]


    def predictNNTemp(self,xs):

        feed_input = {
            self.inputX: xs.astype(np.float64)
        }

        outputVars = [self.nnTemp]
        outputValues = self.sess.run(outputVars,feed_dict=feed_input)

        # print(outputValues)
        return outputValues[0]


    def predictNNTempFirst(self,xs):

        feed_input = {
            self.inputX: xs
        }

        outputVars = [self.nnTempFirst]
        outputValues = self.sess.run(outputVars,feed_dict=feed_input)

        return outputValues[0]


    def trainValues(self,numRepeats,learningRate,xs,ys):

        previousLoss = np.inf

        feed_input = {
            self.inputX: xs,
            self.outputTrue: ys,
            self.lr: learningRate
        }


        pbar = tqdm(range(0,numRepeats))
        for b in pbar:
            _,val_l = self.sess.run([self.train,self.loss],feed_dict=feed_input)
            # print("i: %i loss: %e" % (b,val_l))
            pbar.set_description("loss %e prev: %e" % (val_l,previousLoss))
            previousLoss = val_l


    def getAllWeights(self):

        vars = tf.trainable_variables()
        vars_vals = self.sess.run(vars)

        retDict = {}
        for var, val in zip(vars, vars_vals):
            retDict[var.name] = val
        return retDict


    def printAllWeights(self):
        #https://stackoverflow.com/questions/36193553/get-the-value-of-some-weights-in-a-model-trained-by-tensorflow
        vars = tf.trainable_variables()
        print(vars)
        vars_vals = self.sess.run(vars)

        allVals = []
        allMats = []
        for var, val in zip(vars, vars_vals):
            print("var: {}, value: {}".format(var.name, val))
            if(len(val.shape)==2):
                for i in range(0,val.shape[0]):
                    for j in range(0,val.shape[1]):
                        print(val[i,j],end=" ")
                    print()

            allVals.append([var.name, val])
            allMats.append(Matrix(val))

        #SOME TESTING FUNCTIONS...
        x,y,z = sympy.symbols('x y z')

        for i in range(0,len(allMats)):
            print("i shape:")
            print(allMats[i].shape)

        xyzMat = Matrix([x,y,z])
        print(xyzMat)
        print("xyzMat.shape")
        print(xyzMat.shape)

        m1 = allMats[0].transpose().multiply(xyzMat) + allMats[1]
        print("m1.shape")
        print(m1.shape)

        m1P = m1.multiply_elementwise(m1)
        print("m1P.shape")
        print(m1P.shape)

        m2 = allMats[2].transpose().multiply(m1P) + allMats[3]
        print("m2.shape")
        print(m2.shape)
        for i in range(0,3):
            print(sympy.expand(m2[i]))

        tryVal = [-1,-1,-2]

        print("eval test")
        e = m2.subs([(x,tryVal[0]),(y,tryVal[1]),(z,tryVal[2])])
        for i in range(0,3):
            print(e[i])

        print("eval test 2")
        e2 = m1P.subs([(x,tryVal[0]),(y,tryVal[1]),(z,tryVal[2])])
        for i in range(0,3):
            print(e2[i])


        print("eval test m1")
        e3 = m1.subs([(x,tryVal[0]),(y,tryVal[1]),(z,tryVal[2])])
        for i in range(0,3):
            print(e3[i])


        predCheck = self.predict([tryVal])
        print("predCheck")
        print(predCheck)


        predCheck2 = self.predictNNTemp([tryVal])
        print("predCheck2")
        print(predCheck2)

        predCheckM1 = self.predictNNTempFirst([tryVal])
        print("predCheckM1")
        print(predCheckM1)
