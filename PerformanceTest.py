import numpy as np
import DataGenerator
import GCSS
import LKIF
import PCMCI
import Visualization as vis
#from matplotlib import pyplot as plt
from progress.bar import Bar
import time
import argparse

# returns a numpy array of shape (2, len(algorithms))
# or shape (4, len(algorithms)) for evalType "full" 
def getExecutionTimes(couplingMatrix, algorithms, model, samples, alpha, couplingStrength, noiseScale, tauMax, seeds, deltaTCascadeOutput):
    # dont change autoregressive components
    matrix = couplingStrength * couplingMatrix
    for i in range(matrix.shape[0]):
        matrix[i,i] = couplingMatrix[i,i]
    dataSets = np.zeros((len(seeds), samples, np.array(matrix).shape[0]))
    if model == "Cascade":
        for i in range(len(seeds)):
            dataSets[i] = DataGenerator.getCascadeData(matrix, samples, deltaTCascadeOutput, noiseScale, seeds[i])
    elif model == "VAR":
        for i in range(len(seeds)):
            dataSets[i] = DataGenerator.getVARData(matrix, samples, noiseScale, seeds[i])
    times = []
    # fullData has the shape (observations, variables), LKIF and GCSS want that transposed
    # afer 100 seconds, terminate and count successful executions
    executionsFinished = -1
    if "GCSS" in algorithms:
        timeGCSS = time.process_time_ns()
        for i in range(len(seeds)):
            try:
                matrixGCSS = GCSS.gcss(dataSets[i].T, alpha, tauMax, returnAll=False)
                if time.process_time_ns() - timeGCSS > 100000000000:
                    executionsFinished = i
                    break
            except:
                print("Error on GCSS")
        if executionsFinished >= 0:
            timeGCSS = 100000000000 / np.maximum(1,executionsFinished)
            print("GCSS executed less samples: " + str(executionsFinished))
        else:
            timeGCSS = (time.process_time_ns() - timeGCSS) / len(seeds)
        times.append(timeGCSS)
    executionsFinished = -1
    if "LKIF" in algorithms:
        timeLKIF = time.process_time_ns()
        for i in range(len(seeds)):
            matrixLKIF = LKIF.lkif(dataSets[i].T, alpha)
            if time.process_time_ns() - timeLKIF > 100000000000:
                executionsFinished = i
                break
        if executionsFinished >= 0:
            timeLKIF = 100000000000 / np.maximum(1,executionsFinished)
            print("LKIF executed less samples: " + str(executionsFinished))
        else:
            timeLKIF = (time.process_time_ns() - timeLKIF) / len(seeds)
        times.append(timeLKIF)
    executionsFinished = -1
    if "PCMCI" in algorithms:
        timePCM = time.process_time_ns()
        for i in range(len(seeds)):
            matrixPCMCI, p_values = PCMCI.PCMCIPlus(dataSets[i], [], range(dataSets[i].shape[1]), None, tauMax, alpha if alpha<=1 else 1, contempLinks=True)
            if time.process_time_ns() - timePCM > 100000000000:
                executionsFinished = i
                break
        if executionsFinished >= 0:
            timePCM = 100000000000 / np.maximum(1,executionsFinished)
            print("PCMCI executed less samples: " + str(executionsFinished))
        else:
            timePCM = (time.process_time_ns() - timePCM) / len(seeds)
        times.append(timePCM)
    return np.array(times)

def compareExecutionTimes():
    numVars = [2,3,5,7,10]
    samples = [100,500,1000,10000]
    seeds = range(100)

    if False:
        resultNumVars = np.zeros((len(numVars), 3))
        for numInd in range(len(numVars)):
            print(numInd)
            num = numVars[numInd]
            matrix = np.zeros((num, num))
            resultNumVars[numInd] = getExecutionTimes(matrix, ["GCSS", "LKIF", "PCMCI"], "Cascade", 500, 0.05, 1, 0.1, 5, seeds, 0.1)
        print(resultNumVars)
    # from last execution:
    # PCMCI executed less samples for 10 var system: 44
    resultNumVars = np.array([[1.42187500e+07, 6.25000000e+05, 8.15625000e+07], [1.40625000e+07, 6.25000000e+05, 1.72343750e+08],[1.29687500e+07, 7.81250000e+05, 4.93593750e+08],[1.17187500e+07, 7.81250000e+05, 9.39062500e+08],[1.35937500e+07, 9.37500000e+05, 2.27272727e+09]])
    vis.saveF1Curve((resultNumVars / len(seeds)).T, numVars, "Algorithm Execution Time", "./PerformanceExperiment/perf_numberVars", rowLabels = ["GCSS", "LKIF", "PCMCI"], 
                        xlabel = "System Variables", ylabel = "Execution time in ns", yscale = "log")
    if False:
        resultsSamples = np.zeros((len(samples), 3))
        for sampInd in range(len(samples)):
            print(sampInd)
            sampleCount = samples[sampInd]
            matrix = np.zeros((3, 3))
            resultsSamples[sampInd] = getExecutionTimes(matrix, ["GCSS", "LKIF", "PCMCI"], "Cascade", sampleCount, 0.05, 1, 0.1, 5, seeds, 0.1)
        print(resultsSamples)
    resultsSamples = np.array([[1.8906250e+07, 6.2500000e+05, 1.8187500e+08],[1.5625000e+07, 7.8125000e+05, 2.1890625e+08],[1.5468750e+07, 7.8125000e+05, 2.2625000e+08], [2.0000000e+07, 3.4375000e+06, 9.3140625e+08]])
    vis.saveF1Curve((resultsSamples / len(samples)).T, samples, "Algorithm Execution Time", "./PerformanceExperiment/perf_samples", rowLabels = ["GCSS", "LKIF", "PCMCI"], 
                        xlabel = "Sample Count", ylabel = "Execution time in ns", yscale = "log", xscale="log")

if __name__ == "__main__":
    compareExecutionTimes()
        