#DKE Green
#2019

import numpy as np
import tensorflow as tf



class NNManagerBase():

    def __init__(self,name,inputDim,outputDim):

        self.name = name

        self.inputDim = inputDim
        self.outputDim = outputDim

        self._setupNetwork()
        self._setupTraining()

        self.saver = tf.train.Saver()



    def save(self,path):
        save_path = saver.save(self.sess, path)
        return save_path

    def load(self,path):
        self.saver.restore(self.sess, path)


    def _setupNetwork(self):
        pass

    def _setupTraining(self):
        pass

    def predict(self,xs):
        pass

    def trainValues(self,numRepeats,learningRate,xs,ys):
        pass


    def setSession(self,sess):
        self.sess = sess

    def initSession(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        self.sess = sess
