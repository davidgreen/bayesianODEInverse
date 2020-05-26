#DKE Green
#2019

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from sprottTrainer import SprottTrainer



def main(seed,numTimes,keepProb):

    np.random.seed(seed)
    tf.random.set_random_seed(seed)


    trainer = SprottTrainer()

    numTraces = 1000
    data,dataTest,allOutputDataArray = trainer.generateOutput(numTimes=numTimes,keepProb=keepProb,numTraces=numTraces)


    avgs = np.average(allOutputDataArray,axis=0)
    stds = np.std(allOutputDataArray,axis=0)

    ciFactor = 1.96
    ciHigh = avgs + ciFactor*stds
    ciLow = avgs - ciFactor*stds


    #now, need to write to file.


    plt.figure()
    plt.plot(avgs[:,0],color="orange")
    plt.plot(avgs[:,0]+ciFactor*stds[:,0],color="red")
    plt.plot(avgs[:,0]-ciFactor*stds[:,0],color="red")

    plt.plot(dataTest['data'][:,0],label="x",linewidth=4,color="black")

    # plt.ylim(-10,10)

    # plt.legend()
    plt.show()


    fileName = str(seed) + "-" + str(numTimes) + "-" + str(keepProb) + ".dat"
    with open(fileName, 'w') as f:
        f.write("t avgx avgy avgz ciupx ciupy ciupz cidnx cidny cidnz\n")
        for i in range(0,numTimes):
            nextLine = str(data['times'][i])
            nextLine += " " + str(avgs[i,0])
            nextLine += " " + str(avgs[i,1])
            nextLine += " " + str(avgs[i,2])

            nextLine += " " + str(ciHigh[i,0])
            nextLine += " " + str(ciHigh[i,1])
            nextLine += " " + str(ciHigh[i,2])

            nextLine += " " + str(ciLow[i,0])
            nextLine += " " + str(ciLow[i,1])
            nextLine += " " + str(ciLow[i,2])

            f.write(nextLine + '\n')






if __name__ == "__main__":

    seed = int(sys.argv[1])
    numTimes = int(sys.argv[2])
    keepProb = float(sys.argv[3])

    print("Running seed: %i, numTimes: %i, keepProb: %e" % (seed, numTimes, keepProb))


    main(seed,numTimes,keepProb)
