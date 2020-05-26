# How to run


## To generate data files generally

Run ```mainTrain.py``` with args described below.

```
python3 mainTrain.py ${SEED} ${NUMTIMES} ${KEEPPROB}
```

Data will be written to a file with name:
```
"${SEED}-"${numTimes}-${KEEPPROB}.dat"
```

Command line args are:
1. ${SEED}: the random seed used to generate the data, weights and so on.
2. ${NUMTIMES}: the number of times, between $t=0$ and $t=100$ at which to output a trajectory estimate. The spacing between data points will be $h = \frac{100}{$\lbrace NUMTIMES \rbrace}$.
3. ${KEEPPROB}: The dropout rate, $r$, as described in the paper.


# Command line calls to generate data for each figure.

## To produce training and test traces (data for Figs 1, 2 and 3)

```
python3 mainGenerateTraceAndTrainingData.py
```

## Figure 4

```
python3 mainTrain.py 173451 2500 0.75
python3 mainTrain.py 173456 5000 0.75
python3 mainTrain.py 17345 10000 0.75
```


## Figure 5

```
python3 mainTrain.py 173456 5000 0.75
```


## Figure 6

```
python3 mainTrain.py 16460 10000 0.75
```


## Figure 7

```
python3 mainTrain.py 33333 50000 0.75
python3 mainTrain.py 834576 1000 0.75
python3 mainTrain.py 83457 500 0.75
python3 mainTrain.py 17345 10000 0.75

```




## Figures 8 and 9

```
python3 mainTrain.py 173456 5000 0.75
python3 mainTrain.py 16679 5000 0.95
python3 mainTrain.py 16678 5000 0.5
```
