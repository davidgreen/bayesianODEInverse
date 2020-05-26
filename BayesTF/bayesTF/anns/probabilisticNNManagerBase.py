#DKE Green
#2019

import numpy as np
import tensorflow as tf


from .nnManagerBase import NNManagerBase


class ProbabilisticNNManagerBase(NNManagerBase):

    def __init__(self,name,inputDim,outputDim):
        NNManagerBase.__init__(self,name,inputDim,outputDim)

    def probabilisticPredict(self,xs,numSamples):
        pass
