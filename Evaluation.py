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

# get label encoding the full ranking of 3 algorithms
def getLabel(scoreTriplet):
    gcss, lkif, pcm = scoreTriplet
    if gcss > lkif:
        if pcm > gcss:
            #return "PCMCIGCSSLKIF"
            return 0
        elif pcm > lkif:
            #return "GCSSPCMCILKIF"
            return 1
        else:
            #return "GCSSLKIFPCMCI"
            return 2
    else:
        if pcm > lkif:
            #return "PCMCILKIFGCSS"
            return 3
        elif pcm > gcss:
            #return "LKIFPCMCIGCSS"
            return 4
        else:
            #return "LKIFGCSSPCMCI"
            return 5
        
# get a label that is the highest scoring algorithm for easier visual representation
def getMaxLabel(scoreTriplet):
    gcss, lkif, pcm = scoreTriplet
    if gcss == lkif and lkif == pcm: return 3
    if gcss > lkif:
        if gcss > pcm: return 0
        elif gcss == pcm: return 3
        else: return 2
    elif pcm > lkif: return 2
    elif pcm == lkif or gcss == lkif: return 3
    else: return 1

def getMeanStdDev(data, axis = None):
    return np.average(data, axis = axis), np.std(data, axis = axis)

def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

def maxSignificantLink(x, bool_matr, axis = None):
    y = np.multiply(x, bool_matr)
    return absmaxND(y, axis = axis)

def getFullMetrics(a,b):
    a = np.abs(a) > 0
    b = np.abs(b) > 0
    #convert bool to int
    a = a*1
    b = b*1
    truePos = np.sum(a+b == 2)
    # diagonals are always true negatives, subtract them
    trueNeg = np.sum(a+b == 0) - a.shape[0]
    falsePos = np.sum(np.maximum(b-a,0))
    falseNeg = np.sum(np.maximum(a-b,0))
    return [truePos, trueNeg, falsePos, falseNeg]

def f1Score(a,b):
    a = np.abs(a) > 0
    b = np.abs(b) > 0
    #convert bool to int
    a = a*1
    b = b*1
    truePos = np.sum(a+b == 2)
    falsePos = np.sum(np.maximum(b-a,0))
    falseNeg = np.sum(np.maximum(a-b,0))
    return [float(2 * truePos / (2 * truePos + falsePos + falseNeg)),0]

def f1Score_formula(arr):
    truePos, trueNeg, falsePos, falseNeg = arr
    return float(2 * truePos / (2 * truePos + falsePos + falseNeg))

def f1ScoreFromFull(a, axis=0):
    if a.shape[int(axis)] != 4:
        print("Error: Can't determine MCC from falsely shaped array")
    return np.apply_along_axis(f1Score_formula, axis, a)

def MCC(arr):
    #print(arr)
    truePos, trueNeg, falsePos, falseNeg = arr
    return float(((truePos * trueNeg) - (falsePos * falseNeg)) / np.sqrt((truePos + falsePos) * (truePos + falseNeg) * (trueNeg + falsePos) * (trueNeg + falseNeg)) if (truePos + falsePos) * (truePos + falseNeg) * (trueNeg + falsePos) * (trueNeg + falseNeg) != 0 else 0)

def MCCFromFull(a, axis=0):
    if a.shape[int(axis)] != 4:
        print("Error: Can't determine MCC from falsely shaped array")
    return np.apply_along_axis(MCC, axis, a)

#with ground truth a, result b
def tpr_fpr_Scores(a,b):
    a = np.abs(a) > 0
    b = np.abs(b) > 0
    #convert bool to int
    a = a*1
    b = b*1
    totalPos = np.sum(a)
    # remove diagonal from negative
    totalNeg = a.shape[0]*a.shape[0] - totalPos - a.shape[0]
    truePos = np.sum(a+b == 2)
    falsePos = np.sum(np.maximum(b-a,0))
    tpr = truePos / totalPos
    fpr = falsePos / totalNeg
    return [tpr,fpr]

def tpr_fpr(arr):
    truePos, trueNeg, falsePos, falseNeg = arr
    return [float(truePos / (truePos + falseNeg)), float(falsePos / (falsePos + trueNeg))]

def tpr_fpr_FromFull(a, axis=0):
    if a.shape[int(axis)] != 4:
        print("Error: Can't determine MCC from falsely shaped array")
    return np.apply_along_axis(tpr_fpr, axis, a)

# returns a numpy array of shape (2, len(algorithms))
# or shape (4, len(algorithms)) for evalType "full" 
def getMetricOfRealization(couplingMatrix, algorithms, model, samples, alpha, couplingStrength, noiseScale, tauMax, seed, deltaTCascadeOutput, evalType):
    # dont change autoregressive components
    matrix = couplingStrength * couplingMatrix
    for i in range(matrix.shape[0]):
        matrix[i,i] = couplingMatrix[i,i]
    if model == "Cascade":
        fullData = DataGenerator.getCascadeData(matrix, samples, deltaTCascadeOutput, noiseScale, seed)
    elif model == "VAR":
        fullData = DataGenerator.getVARData(matrix, samples, noiseScale, seed)
    matrices = []
    # fullData has the shape (observations, variables), LKIF and GCSS want that transposed
    if "GCSS" in algorithms:
        try:
            matrixGCSS = GCSS.gcss(fullData.T, alpha, tauMax, returnAll=False)
            matrixGCSS[np.isnan(matrixGCSS)] = 0
        except:
            matrixGCSS = np.zeros((fullData.shape[1],fullData.shape[1]))
            print("Error on GCSS")
        np.fill_diagonal(matrixGCSS,0)
        matrices.append(matrixGCSS)
    if "LKIF" in algorithms:
        matrixLKIF = LKIF.lkif(fullData.T, alpha)
        np.fill_diagonal(matrixLKIF, 0)
        matrices.append(matrixLKIF.T)
    if "PCMCI" in algorithms:
        matrixPCMCI, p_values = PCMCI.PCMCIPlus(fullData, [], range(fullData.shape[1]), None, tauMax, alpha if alpha<=1 else 1, contempLinks=True)
        graph_bool = p_values <= alpha
        matrixPCMCI = maxSignificantLink(matrixPCMCI, graph_bool, axis = 2)
        np.fill_diagonal(matrixPCMCI,0)
        matrices.append(matrixPCMCI.T)
    np.fill_diagonal(couplingMatrix, 0)
    if evalType == "ROC":
        metrics = np.array([tpr_fpr_Scores(couplingMatrix, matrix) for matrix in matrices])
    elif evalType == "F1":
        metrics = np.array([f1Score(couplingMatrix, matrix) for matrix in matrices])
    elif evalType == "Full":
        metrics = np.array([getFullMetrics(couplingMatrix, matrix) for matrix in matrices])
    else:
        print("Error: Invalid evaluation type found")
        exit()
    return metrics.T

# if iterateSeeds is set to false and evaluateSeparately is set to true, the same seed is used for data generation with varying parameter values
def runAccuracyEvaluation(algorithms, model, evaluateSeparately, defaultCouplingMatrix, defaultSamples, defaultAlpha, defaultCouplingStrength, defaultNoiseScale, defaultTauMax,
                  couplingMatrices, samples, alphas, couplingStrengths, noiseScales, tauMaxs, runsPerConfig, startSeed = 0, deltaTCascadeOutput = 0.1, fakeRun = False, evalType = "ROC",
                  iterateSeeds = True):
    validAlgorithms = ["PCMCI", "LKIF", "GCSS"]
    validModels = ["VAR", "Cascade"]
    assert len(couplingMatrices) + len(samples) + len(alphas) + len(couplingStrengths) + len(noiseScales) + len(tauMaxs) > 0
    assert len(algorithms) > 0 and np.all(algorithm in validAlgorithms for algorithm in algorithms)
    assert model in validModels

    metricsLength = 4 if evalType == "Full" else 2
    if evaluateSeparately:
        runs = runsPerConfig * (len(couplingMatrices) + len(samples) + len(alphas) + len(couplingStrengths) + len(noiseScales) + len(tauMaxs))
        currentSeed = startSeed
        progressBar = Bar('Processing', max=runs)
        matricesResults = np.zeros((metricsLength, len(algorithms), runsPerConfig, len(couplingMatrices)))
        sampleResults = np.zeros((metricsLength, len(algorithms), runsPerConfig, len(samples)))
        couplingStrengthResults = np.zeros((metricsLength, len(algorithms), runsPerConfig, len(couplingStrengths)))
        noiseScaleResults = np.zeros((metricsLength, len(algorithms), runsPerConfig, len(noiseScales)))
        alphaResults = np.zeros((metricsLength, len(algorithms), runsPerConfig, len(alphas)))
        tauMaxResults = np.zeros((metricsLength, len(algorithms), runsPerConfig, len(tauMaxs)))
        if not fakeRun:
            for i in range(runsPerConfig):
                if not iterateSeeds:
                    currentSeed += 1
                for matrixIndex in range(len(couplingMatrices)):
                    couplingMatrix = couplingMatrices[matrixIndex]
                    matricesResults[:,:,i,matrixIndex] = getMetricOfRealization(couplingMatrix, algorithms, model, defaultSamples, defaultAlpha, defaultCouplingStrength, defaultNoiseScale, defaultTauMax, currentSeed, deltaTCascadeOutput, evalType)
                    if iterateSeeds:
                        currentSeed += 1
                    progressBar.next()
                for sampleIndex in range(len(samples)):
                    sample = samples[sampleIndex]
                    sampleResults[:,:,i,sampleIndex] = getMetricOfRealization(defaultCouplingMatrix, algorithms, model, sample, defaultAlpha, defaultCouplingStrength, defaultNoiseScale, defaultTauMax, currentSeed, deltaTCascadeOutput, evalType)
                    if iterateSeeds:
                        currentSeed += 1
                    progressBar.next()
                for alphaIndex in range(len(alphas)):
                    alpha = alphas[alphaIndex]
                    alphaResults[:,:,i,alphaIndex]=getMetricOfRealization(defaultCouplingMatrix, algorithms, model, defaultSamples, alpha, defaultCouplingStrength, defaultNoiseScale, defaultTauMax, currentSeed, deltaTCascadeOutput, evalType)
                    if iterateSeeds:
                        currentSeed += 1
                    progressBar.next()
                for couplingStrengthIndex in range(len(couplingStrengths)):
                    couplingStrength = couplingStrengths[couplingStrengthIndex]
                    couplingStrengthResults[:,:,i,couplingStrengthIndex]=getMetricOfRealization(defaultCouplingMatrix, algorithms, model, defaultSamples, defaultAlpha, couplingStrength, defaultNoiseScale, defaultTauMax, currentSeed, deltaTCascadeOutput, evalType)
                    if iterateSeeds:
                        currentSeed += 1
                    progressBar.next()
                for noiseScaleIndex in range(len(noiseScales)):
                    noiseScale = noiseScales[noiseScaleIndex]
                    noiseScaleResults[:,:,i,noiseScaleIndex] = getMetricOfRealization(defaultCouplingMatrix, algorithms, model, defaultSamples, defaultAlpha, defaultCouplingStrength, noiseScale, defaultTauMax, currentSeed, deltaTCascadeOutput, evalType)
                    if iterateSeeds:
                        currentSeed += 1
                    progressBar.next()
                for tauMaxIndex in range(len(tauMaxs)):
                    tauMax = tauMaxs[tauMaxIndex]
                    tauMaxResults[:,:,i,tauMaxIndex]=getMetricOfRealization(defaultCouplingMatrix, algorithms, model, defaultSamples, defaultAlpha, defaultCouplingStrength, defaultNoiseScale, tauMax, currentSeed, deltaTCascadeOutput, evalType)
                    if iterateSeeds:
                        currentSeed += 1
                    progressBar.next()
            progressBar.finish()
        return currentSeed, [matricesResults, sampleResults, alphaResults, couplingStrengthResults, noiseScaleResults, tauMaxResults]
    else:
        if len(couplingMatrices) == 0:
            couplingMatrices = [defaultCouplingMatrix]
        if len(samples) == 0:
            samples = [defaultSamples]
        if len(alphas) == 0:
            alphas = [defaultAlpha]
        if len(couplingStrengths) == 0:
            couplingStrengths = [defaultCouplingStrength]
        if len(noiseScales) == 0:
            noiseScales = [defaultNoiseScale]
        if len(tauMaxs) == 0:
            tauMaxs = [defaultTauMax]
        runs = runsPerConfig * len(couplingMatrices) * len(samples) * len(alphas) * len(couplingStrengths) * len(noiseScales) * len(tauMaxs)
        results = np.zeros((metricsLength, len(algorithms), runsPerConfig, len(couplingMatrices), len(samples), len(alphas), len(couplingStrengths), len(noiseScales), len(tauMaxs)))
        currentSeed = startSeed
        progressBar = Bar('Processing', max=runs)
        if not fakeRun:
            for matrixIndex in range(len(couplingMatrices)):
                couplingMatrix = couplingMatrices[matrixIndex]
                for sampleIndex in range(len(samples)):
                    sample = samples[sampleIndex]
                    for alphaIndex in range(len(alphas)):
                        alpha = alphas[alphaIndex]
                        for couplingStrengthIndex in range(len(couplingStrengths)):
                            couplingStrength = couplingStrengths[couplingStrengthIndex]
                            for noiseScaleIndex in range(len(noiseScales)):
                                noiseScale = noiseScales[noiseScaleIndex]
                                for tauMaxIndex in range(len(tauMaxs)):
                                    tauMax = tauMaxs[tauMaxIndex]
                                    for i in range(runsPerConfig):
                                        results[:,:,i,matrixIndex,sampleIndex,alphaIndex,couplingStrengthIndex,noiseScaleIndex,tauMaxIndex] = getMetricOfRealization(couplingMatrix, algorithms, model, sample, alpha, couplingStrength, noiseScale, tauMax, currentSeed, deltaTCascadeOutput, evalType)
                                        currentSeed += 1
                                        progressBar.next()
            progressBar.finish()
        return currentSeed, results
    
def mainEvaluations(analysisIndex = None):

    startTime = time.process_time()

    folder = "./MainExperiment/"

    
    # since a and b in the x^3 equations make up the autoregressive component, we replace it by 0.5 for var systems
    defaultCouplingMatrixVAR= np.array([[0.5,1,0],[-1,0.5,0],[1,-1,0.5]])
    defaultCouplingMatrixCascade = np.array([[0,1,0],[-1,0,0],[1,-1,0]])
    defaultSamples = 500
    defaultCouplingStrengthVAR = 0.1
    defaultCouplingStrengthCascade = 1
    defaultNoiseScale = 0.1
    defaultTauMax = 2

        
    # larger systems, similar density - about 1.3 edges per vertex
    mediumCouplingMatrixVAR = np.array([[0.5,0,-1,1,0,0],
                                        [1,0.5,0,0,0,0],
                                        [0,1,0.5,0,0,0],
                                        [0,0,0,0.5,0,0],
                                        [0,0,1,-1,0.5,1],
                                        [0,0,0,0,-1,0.5]])
    mediumCouplingMatrixCascade = np.array([[0,0,-1,1,0,0],
                                        [1,0,0,0,0,0],
                                        [0,1,0,0,0,0],
                                        [0,0,0,0,0,0],
                                        [0,0,1,-1,0,1],
                                        [0,0,0,0,-1,0]])
            
    largeCouplingMatrixVAR = np.array([[0.5,0,-1,0,0,1,0,0,0,0,0,0],
                                        [1,0.5,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0.5,0,0,0,0,1,0,0,0,0],
                                        [0,1,0,0.5,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0.5,-1,0,0,0,0,0,0],
                                        [0,0,0,-1,0,0.5,0,0,0,0,0,0],
                                        [0,0,-1,0,0,1,0.5,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0.5,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0.5,1,0,0],
                                        [0,0,0,0,1,0,0,0,0,0.5,0,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0.5,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0.5]])
    largeCouplingMatrixCascade = np.array([[0,0,-1,0,0,1,0,0,0,0,0,0],
                                        [1,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,1,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,-1,0,0,0,0,0,0],
                                        [0,0,0,-1,0,0,0,0,0,0,0,0],
                                        [0,0,-1,0,0,1,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0,1,0,0],
                                        [0,0,0,0,1,0,0,0,0,0,0,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0]])
    
    mediumCouplingMatrixCascade_LowDense = np.array([
                                        [0,0,0,0,0,0],
                                        [1,0,0,0,0,0],
                                        [0,-1,0,0,0,0],
                                        [0,0,0,0,0,0],
                                        [0,0,1,0,0,1],
                                        [0,0,0,0,-1,0]])
    mediumCouplingMatrixVAR_LowDense = np.array([
                                        [0.5,0,0,0,0,0],
                                        [1,0.5,0,0,0,0],
                                        [0,-1,0.5,0,0,0],
                                        [0,0,0,0.5,0,0],
                                        [0,0,1,0,0.5,1],
                                        [0,0,0,0,-1,0.5]])
    
    mediumCouplingMatrixVAR_HighDense = np.array([
                                        [0.5,0,-1,-1,0,0],
                                        [1,0.5,0,0,0,0],
                                        [1,1,0.5,0,0,0],
                                        [0,0,0,0.5,-1,-1],
                                        [0,1,-1,0,0.5,1],
                                        [0,0,0,1,-1,0.5]])
    mediumCouplingMatrixCascade_HighDense = np.array([
                                        [0,0,-1,-1,0,0],
                                        [1,0,0,0,0,0],
                                        [1,1,0,0,0,0],
                                        [0,0,0,0,-1,-1],
                                        [0,1,-1,0,0,1],
                                        [0,0,0,1,-1,0]])
    
    defaultCouplingMatrixVAR_LowDense= np.array([[0.5,0,0],[-1,0.5,0],[0,-1,0.5]])
    defaultCouplingMatrixCascade_LowDense = np.array([[0,0,0],[-1,0,0],[0,-1,0]])

    defaultCouplingMatrixVAR_HighDense= np.array([[0.5,1,0],[-1,0.5,1],[1,-1,0.5]])
    defaultCouplingMatrixCascade_HighDense = np.array([[0,1,0],[-1,0,1],[1,-1,0]])

    largeCouplingMatrixVAR_LowDense = np.array([[0.5,0,0,0,0,0,0,0,0,0,0,0],
                                        [1,0.5,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0.5,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0.5,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0.5,-1,0,0,0,0,0,0],
                                        [0,0,0,-1,0,0.5,0,0,0,0,0,0],
                                        [0,0,-1,0,0,0,0.5,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0.5,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0.5,0,0,0],
                                        [0,0,0,0,1,0,0,0,0,0.5,0,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0.5,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0.5]])
    largeCouplingMatrixCascade_LowDense = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                                        [1,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,-1,0,0,0,0,0,0],
                                        [0,0,0,-1,0,0,0,0,0,0,0,0],
                                        [0,0,-1,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,1,0,0,0,0,0,0,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0]])
    largeCouplingMatrixVAR_HighDense = np.array([[0.5,0,-1,0,0,1,0,0,0,0,0,0],
                                        [1,0.5,-1,0,0,0,0,0,0,0,0,0],
                                        [0,1,0.5,0,0,0,0,1,0,0,0,0],
                                        [0,1,0,0.5,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,0.5,-1,0,0,-1,-1,0,0],
                                        [0,0,0,-1,0,0.5,0,0,0,0,0,0],
                                        [0,0,-1,0,0,1,0.5,0,0,0,0,0],
                                        [0,0,0,0,0,0,-1,0.5,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0.5,1,0,0],
                                        [0,0,0,0,1,0,0,0,0,0.5,1,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0.5,1],
                                        [0,0,0,1,0,0,0,0,0,0,0,0.5]])
    largeCouplingMatrixCascade_HighDense = np.array([[0,0,-1,0,0,1,0,0,0,0,0,0],
                                        [1,0,-1,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,1,0,0,0,0],
                                        [0,1,0,0,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,0,-1,0,0,-1,-1,0,0],
                                        [0,0,0,-1,0,0,0,0,0,0,0,0],
                                        [0,0,-1,0,0,1,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,-1,0,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0,1,0,0],
                                        [0,0,0,0,1,0,0,0,0,0,1,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0,1],
                                        [0,0,0,1,0,0,0,0,0,0,0,0]])
    matrixFolder = "./MatrixGraphs/"
    saveCouplingMatrices = True
    checkCouplingMatrices1 = False
    checkCouplingMatrices2 = False
    showCouplingMatrices = False
    if checkCouplingMatrices1:
        vis.saveCouplingMatrixGraph(defaultCouplingMatrixVAR, "3 Vars, Medium Density", matrixFolder+"SmallMedDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(defaultCouplingMatrixVAR_HighDense, "3 Vars, High Density", matrixFolder+"SmallHighDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(defaultCouplingMatrixVAR_LowDense, "3 Vars, Low Density", matrixFolder+"SmallLowDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(mediumCouplingMatrixVAR, "6 Vars, Medium Density", matrixFolder+"MedMedDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(mediumCouplingMatrixVAR_HighDense, "6 Vars, High Density", matrixFolder+"MedHighDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(mediumCouplingMatrixVAR_LowDense, "6 Vars, Low Density", matrixFolder+"MedLowDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(largeCouplingMatrixVAR, "12 Vars, Medium Density", matrixFolder+"LargeMedDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(largeCouplingMatrixVAR_HighDense, "12 Vars, High Density", matrixFolder+"LargeHighDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(largeCouplingMatrixVAR_LowDense, "12 Vars, Low Density", matrixFolder+"LargeLowDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
    if checkCouplingMatrices2:
        vis.saveCouplingMatrixGraph(defaultCouplingMatrixVAR, "", matrixFolder+"SmallMedDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(defaultCouplingMatrixVAR_HighDense, "", matrixFolder+"SmallHighDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(defaultCouplingMatrixVAR_LowDense, "", matrixFolder+"SmallLowDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(mediumCouplingMatrixVAR, "", matrixFolder+"MedMedDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(mediumCouplingMatrixVAR_HighDense, "", matrixFolder+"MedHighDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(mediumCouplingMatrixVAR_LowDense, "", matrixFolder+"MedLowDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(largeCouplingMatrixVAR, "", matrixFolder+"LargeMedDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(largeCouplingMatrixVAR_HighDense, "", matrixFolder+"LargeHighDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(largeCouplingMatrixVAR_LowDense, "", matrixFolder+"LargeLowDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
    #exit()

    # about 20 minutes? Felt more like 40, maybe windows scheduler messed this up?
    runsPerConfigAlpha = 10
    runsPerConfigTauMax = 20
    # about 2 minutes
    runsPerConfigSeparateParams = 20
    runsPerConfigCombinedParams = 10
    runsPerConfigAlphaCombination = 10
    runsPerConfigAlphaBonus = 20
    doSave = True
    showDiagrams = False
    # to test that data shapes are correctly transmitted everywhere
    fakeRun = False

    plotAlpha = (True and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 1
    calculateAlpha = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 1
    alphaSizeFileName = "alphaSizeAnalysis.npy"
    alphaDensityFileName = "alphaDensityAnalysis.npy"

    plotTauMax = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 2
    calculateTauMax = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 2
    tauMaxFileName = "tauMaxAnalysis.npy"

    plotSeparateDataParameters = (True and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 3
    calculateSeparateDataParameters = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 3
    dataParamsFileName = "dataParameters.npz"

    plotDecisionTree = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 4
    calculateFullSearchSpace = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 4
    fullSearchFileName = "fullSearchSpace.npy"

    plotAlphaCombined = (True and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 5
    calculateAlphaCombined = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 5
    alphaCombinedFileName = "alphaCombination.npy"

    plotBonusAlpha = (True and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 6
    calculateBonusAlpha = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 6
    calculateBonusAlpha2 = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 6

    seed = 0
    alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    tauMaxs = [2,3,5,7,10]

    samples_separate = [20,50,100,200,500,1000,2000,5000,10000]
    couplingStrVAR_separate = np.array([0.01, 0.02, 0.05, 0.07,0.1,0.15,0.2,0.25,0.3])
    couplingStrCascade_separate = 10*couplingStrVAR_separate
    noiseScales_separate = [0.01,0.02,0.05,0.1,0.2,0.5]
    # WHENEVER these are changed, make sure to change the plotting logic too!
    matrices_separate_VAR = [defaultCouplingMatrixVAR_LowDense, defaultCouplingMatrixVAR, defaultCouplingMatrixVAR_HighDense, 
                            mediumCouplingMatrixVAR_LowDense, mediumCouplingMatrixVAR, mediumCouplingMatrixVAR_HighDense, 
                            largeCouplingMatrixVAR_LowDense, largeCouplingMatrixVAR, largeCouplingMatrixVAR_HighDense]
    matrices_separate_Cascade = [defaultCouplingMatrixCascade_LowDense, defaultCouplingMatrixCascade, defaultCouplingMatrixCascade_HighDense,
                                 mediumCouplingMatrixCascade_LowDense, mediumCouplingMatrixCascade, mediumCouplingMatrixCascade_HighDense,
                                 largeCouplingMatrixCascade_LowDense, largeCouplingMatrixCascade, largeCouplingMatrixCascade_HighDense]
    
    alphas_combined = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    samples_combined = [100, 500, 1000, 10000]
    couplingStrVAR_combined = np.array([0.01, 0.05, 0.1, 0.3])
    #couplingStrVAR_combined = np.array([0.1])
    couplingStrCascade_combined = 10*couplingStrVAR_combined
    noiseScales_combined = [0.01,0.05, 0.1, 0.5]
    #noiseScales_combined = [0.1]
    matrices_combined_VAR = [defaultCouplingMatrixVAR_LowDense, defaultCouplingMatrixVAR, defaultCouplingMatrixVAR_HighDense, 
                            mediumCouplingMatrixVAR_LowDense, mediumCouplingMatrixVAR, mediumCouplingMatrixVAR_HighDense]
    matrices_combined_Cascade = [defaultCouplingMatrixCascade_LowDense, defaultCouplingMatrixCascade, defaultCouplingMatrixCascade_HighDense,
                                 mediumCouplingMatrixCascade_LowDense, mediumCouplingMatrixCascade, mediumCouplingMatrixCascade_HighDense]

    # Alpha: 2 (VAR/Cascade) * 5 (number of matrices) * len(alphas) * runsPerConfig
    # TauMax: 2 (VAR/Cascade) * len(tauMaxs) * runsPerConfig
    # separate Parameters: 2 (VAR/Cascade) * runsPerConfig * (samples + couplingStr + noiseScales + matrices)
    # combined Parameters: 2(VAR/Cascade) * runsPerConfig * samples * couplingStr * noiseScales * matrices
    executionEstimation = (runsPerConfigAlpha * (10 * len(alphas) if calculateAlpha else 0) 
                                           + (runsPerConfigTauMax * 2 * len(tauMaxs) if calculateTauMax else 0)
                                           + (runsPerConfigSeparateParams * 2 * (len(samples_separate)+ len(couplingStrVAR_separate)+len(noiseScales_separate)+len(matrices_separate_VAR)) if calculateSeparateDataParameters else 0)
                                           + (runsPerConfigCombinedParams * 2 * len(samples_combined) * len(couplingStrVAR_combined) * len(noiseScales_combined) * len(matrices_combined_VAR) if calculateFullSearchSpace else 0))
    print("total runs, roughly equal to number of seconds: ")
    print(executionEstimation)
    print("Minutes estimate: ")
    print(executionEstimation/60)

    if plotAlpha:
        if calculateAlpha:
            #default analysis for 3 vertices, 4 edges
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")

            seed, resultsMediumSize_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, resultsMediumSize_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")
            
            seed, resultsLargeSize_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= largeCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, resultsLargeSize_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= largeCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")
            
            seed, resultsLowDense_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixVAR_LowDense, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, resultsLowDense_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixCascade_LowDense, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")
            
            seed, resultsLargeDense_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixVAR_HighDense, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, resultsLargeDense_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixCascade_HighDense, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")

            # alpha lies in axis 2
            results_VAR = results_VAR[2]
            results_Cascade = results_Cascade[2]
            resultsMediumSize_VAR = resultsMediumSize_VAR[2]
            resultsMediumSize_Cascade = resultsMediumSize_Cascade[2]
            resultsLargeSize_VAR = resultsLargeSize_VAR[2]
            resultsLargeSize_Cascade = resultsLargeSize_Cascade[2]
            resultsLowDense_VAR = resultsLowDense_VAR[2]
            resultsLowDense_Cascade = resultsLowDense_Cascade[2]
            resultsLargeDense_VAR = resultsLargeDense_VAR[2]
            resultsLargeDense_Cascade = resultsLargeDense_Cascade[2]

            resultsSmall = np.stack((results_VAR, results_Cascade), axis=0)
            resultsMedium = np.stack((resultsMediumSize_VAR, resultsMediumSize_Cascade), axis=0)
            resultsLarge = np.stack((resultsLargeSize_VAR, resultsLargeSize_Cascade), axis = 0)
            resultsLowDense = np.stack((resultsLowDense_VAR,resultsLowDense_Cascade),axis=0)
            resultsHighDense = np.stack((resultsLargeDense_VAR,resultsLargeDense_Cascade), axis=0)
            resultsSize = np.stack((resultsSmall, resultsMedium, resultsLarge),axis=0)
            resultsDense = np.stack((resultsLowDense, resultsMedium, resultsHighDense), axis=0)
            if doSave:
                np.save(folder + alphaSizeFileName, resultsSize)
                np.save(folder+alphaDensityFileName, resultsDense)
        else: 
            resultsSize = np.load(folder + alphaSizeFileName)
            resultsDense = np.load(folder+alphaDensityFileName)
        # shape: 3 (low/medium/large size or density), 2 (VAR/Cascade), 2 (TPR/FPR) or 4 (TP/TN/FP/FN), 3 (Algorithm), runsPerConfig, alpha values
        # average/median over runs
        print(np.squeeze(resultsSize[0,0,:,2,:,10]))
        resultsSize = np.average(tpr_fpr_FromFull(resultsSize, axis= 2), axis= 4)
        resultsDense = np.average(tpr_fpr_FromFull(resultsDense, axis= 2), axis=4)
        # visualization for default configuration with 3 vertices, 4 edges
        vis.saveROCCurve(resultsSize[0,:,0,0,:], resultsSize[0,:,1,0,:], alphas, "GCSS ROC Curve for alpha values", folder+"GCSS_Alpha_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsSize[0,:,0,1,:], resultsSize[0,:,1,1,:], alphas, "LKIF ROC Curve for alpha values", folder+"LKIF_Alpha_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsSize[0,:,0,2,:], resultsSize[0,:,1,2,:], alphas, "PCMCI ROC Curve for alpha values", folder+"PCMCI_Alpha_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")

        # for ease of notation: reshape
        resultsSize = np.reshape(resultsSize,(6, resultsSize.shape[2], resultsSize.shape[3],resultsSize.shape[4]),order="F")
        vis.saveROCCurve(resultsSize[:,0,0,:], resultsSize[:,1,0,:], alphas, "GCSS ROC Curve for system sizes", folder+"GCSS_Alpha_Size", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        vis.saveROCCurve(resultsSize[:,0,1,:], resultsSize[:,1,1,:], alphas, "LKIF ROC Curve for system sizes", folder+"LKIF_Alpha_Size", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        vis.saveROCCurve(resultsSize[:,0,2,:], resultsSize[:,1,2,:], alphas, "PCMCI ROC Curve for system sizes", folder+"PCMCI_Alpha_Size", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        
        # for ease of notation: reshape
        resultsDense = np.reshape(resultsDense,(6, resultsDense.shape[2], resultsDense.shape[3],resultsDense.shape[4]),order="F")
        vis.saveROCCurve(resultsDense[:,0,0,:], resultsDense[:,1,0,:], alphas, "GCSS ROC Curve for system densities", folder+"GCSS_Alpha_Density", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        vis.saveROCCurve(resultsDense[:,0,1,:], resultsDense[:,1,1,:], alphas, "LKIF ROC Curve for system densities", folder+"LKIF_Alpha_Density", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        vis.saveROCCurve(resultsDense[:,0,2,:], resultsDense[:,1,2,:], alphas, "PCMCI ROC Curve for system densities", folder+"PCMCI_Alpha_Density", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)

    if plotTauMax:
        if calculateTauMax:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=[],couplingStrengths=[],noiseScales=[],tauMaxs=tauMaxs,
                                            runsPerConfig=runsPerConfigTauMax, fakeRun=fakeRun, startSeed= seed, iterateSeeds=True, evalType="Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=[],couplingStrengths=[],noiseScales=[],tauMaxs=tauMaxs,
                                            runsPerConfig=runsPerConfigTauMax, fakeRun=fakeRun, startSeed=seed, iterateSeeds=True, evalType="Full")
            #tau max in dim 5
            resultsTauMax = np.stack((results_VAR[5], results_Cascade[5]),axis=0)
            if doSave:
                np.save(folder+  tauMaxFileName, resultsTauMax)
        else:
            resultsTauMax = np.load(folder+tauMaxFileName)
        # shape: 2 (VAR/Cascade), 2 (TPR/FPR) or 4 (TP/TN/FP/FN), 2 (Algorithm), runsPerConfig, tauMax values
        resultsTauMax = np.average(tpr_fpr_FromFull(resultsTauMax, axis=1), axis= 3)
        vis.saveROCCurve(resultsTauMax[:,0,0,:], resultsTauMax[:,1,0,:], tauMaxs, "GCSS ROC Curve for tauMax values", folder+"GCSS_tauMax_ConstSeed_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave)
        vis.saveROCCurve(resultsTauMax[:,0,1,:], resultsTauMax[:,1,1,:], tauMaxs, "PCMCI ROC Curve for tauMax values", folder+"PCMCI_tauMax_ConstSeed_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave)
        
    # uses F1 score from here, so matrices are shaped differently!
    if plotSeparateDataParameters:
        if calculateSeparateDataParameters:
            seed, [matricesResults_VAR, sampleResults_VAR, _, couplingStrengthResults_VAR, noiseScaleResults_VAR, _] = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = matrices_separate_VAR, samples=samples_separate,alphas=[],couplingStrengths=couplingStrVAR_separate,noiseScales=noiseScales_separate,tauMaxs=[],
                                            runsPerConfig=runsPerConfigSeparateParams, fakeRun=fakeRun, startSeed= seed, evalType = "Full")
            seed, [matricesResults_Cascade, sampleResults_Cascade, _, couplingStrengthResults_Cascade, noiseScaleResults_Cascade, _] = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = matrices_separate_Cascade, samples=samples_separate,alphas=[],couplingStrengths=couplingStrCascade_separate,noiseScales=noiseScales_separate,tauMaxs=[],
                                            runsPerConfig=runsPerConfigSeparateParams, fakeRun=fakeRun, startSeed=seed, evalType = "Full")
            # save everything regardless of what kind of metric is used later
            matricesResults = np.stack((matricesResults_VAR, matricesResults_Cascade), axis=0)
            sampleResults = np.stack((sampleResults_VAR,sampleResults_Cascade), axis=0)
            couplingStrengthResults = np.stack((couplingStrengthResults_VAR,couplingStrengthResults_Cascade), axis=0)
            noiseScaleResults = np.stack((noiseScaleResults_VAR,noiseScaleResults_Cascade), axis=0)
            if doSave:
                np.savez(folder + dataParamsFileName, matricesResults = matricesResults, sampleResults = sampleResults, couplingStrengthResults = couplingStrengthResults, noiseScaleResults = noiseScaleResults)
        else:
            npzfile = np.load(folder+dataParamsFileName)
            matricesResults = npzfile["matricesResults"]
            sampleResults = npzfile["sampleResults"]
            couplingStrengthResults = npzfile["couplingStrengthResults"] 
            noiseScaleResults = npzfile["noiseScaleResults"]
        # shape: 2 (VAR/Cascade), usually 4 (TP/TN/FP/FN) or previously non-existant axis for F1 scores, 3 (Algorithm), runsPerConfig, iterated values
        matricesResults = MCCFromFull(matricesResults, axis=1)
        sampleResults = MCCFromFull(sampleResults, axis=1)
        couplingStrengthResults = MCCFromFull(couplingStrengthResults, axis=1)
        noiseScaleResults = MCCFromFull(noiseScaleResults, axis=1)
        matricesResults, matricesError = getMeanStdDev(matricesResults, axis = 2)
        sampleResults, sampleError = getMeanStdDev(sampleResults, axis = 2)
        couplingStrengthResults, couplingStrengthError = getMeanStdDev(couplingStrengthResults, axis = 2)
        noiseScaleResults, noiseScaleError = getMeanStdDev(noiseScaleResults, axis = 2)

        #plot samples
        vis.saveF1Curve(sampleResults[0,:,:], samples_separate, "MCC by sample size - VAR systems", folder + "Samples_VAR", errors=sampleError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xscale = "log", xlabel="Samples", ylabel="MCC")
        vis.saveF1Curve(sampleResults[1,:,:], samples_separate, "MCC by sample size - x³ systems", folder + "Samples_Cascade", errors=sampleError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xscale = "log", xlabel="Samples", ylabel="MCC")

        #plot couplingStrengths
        vis.saveF1Curve(couplingStrengthResults[0,:,:], couplingStrVAR_separate, "MCC by coupling strength - VAR systems", folder + "CouplingStrength_VAR",errors=couplingStrengthError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xlabel="Coupling Strength", ylabel="MCC")
        vis.saveF1Curve(couplingStrengthResults[1,:,:], couplingStrCascade_separate, "MCC by coupling strength - x³ systems", folder + "CouplingStrength_Cascade", errors=couplingStrengthError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xlabel="Coupling Strength", ylabel="MCC")

        vis.saveF1Curve(sampleResults[0,:,:], samples_separate, "MCC by sample size - VAR systems", folder + "Poster_Samples_VAR", xlabel="Number of Samples", ylabel="MCC", figsize=(4.5,3.3), dpi=300, errors=sampleError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xscale = "log")
        vis.saveF1Curve(sampleResults[1,:,:], samples_separate, "MCC by sample size - x³ systems", folder + "Poster_Samples_Cascade", xlabel="Number of Samples", ylabel="MCC", figsize=(4.5,3.3), dpi=300, errors=sampleError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xscale = "log")

        #plot couplingStrengths
        vis.saveF1Curve(couplingStrengthResults[0,:,:], couplingStrVAR_separate, "MCC by coupling strength - VAR systems", folder + "Poster_CouplingStrength_VAR",xlabel="Coupling Strength",ylabel="MCC",figsize=(4.5,3.3), dpi=300, errors=couplingStrengthError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave)
        vis.saveF1Curve(couplingStrengthResults[1,:,:], couplingStrCascade_separate, "MCC by coupling strength - x³ systems", folder + "Poster_CouplingStrength_Cascade", xlabel="Coupling Strength",ylabel="MCC",figsize=(4.5,3.3), dpi=300, errors=couplingStrengthError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave)


        #plot noiseScale
        vis.saveF1Curve(noiseScaleResults[0,:,:], noiseScales_separate, "MCC by noise scale - VAR systems", folder + "Noise_VAR",errors=noiseScaleError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xlabel="Noise Scale", ylabel="MCC")
        vis.saveF1Curve(noiseScaleResults[1,:,:], noiseScales_separate, "MCC by noise scale - x³ systems", folder + "Noise_Cascade", errors=noiseScaleError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xlabel="Noise Scale", ylabel="MCC")

        #plot matrices
        matricesResults = np.reshape(matricesResults, (2,3,3,3), order="C")
        matricesError = np.reshape(matricesError, (2,3,3,3), order="C")
        vis.saveF1Curve(matricesResults[0,0,:,:], [3,6,12], "GCSS - MCC by number of variables \n VAR systems", folder + "GCSS_Matrices_VAR", errors=matricesError[0,0,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        vis.saveF1Curve(matricesResults[0,1,:,:], [3,6,12], "LKIF - MCC by number of variables \n VAR systems", folder + "LKIF_Matrices_VAR", errors=matricesError[0,1,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        vis.saveF1Curve(matricesResults[0,2,:,:], [3,6,12], "PCMCI - MCC by number of variables \n VAR systems", folder + "PCMCI_Matrices_VAR", errors=matricesError[0,2,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        
        vis.saveF1Curve(matricesResults[1,0,:,:], [3,6,12], "GCSS - MCC by number of variables \n x³ systems", folder + "GCSS_Matrices_Cascade", errors=matricesError[1,0,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        vis.saveF1Curve(matricesResults[1,1,:,:], [3,6,12], "LKIF - MCC by number of variables \n x³ systems", folder + "LKIF_Matrices_Cascade", errors=matricesError[1,1,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        vis.saveF1Curve(matricesResults[1,2,:,:], [3,6,12], "PCMCI - MCC by number of variables \n x³ systems", folder + "PCMCI_Matrices_Cascade", errors=matricesError[1,2,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")

    if plotDecisionTree:
        if calculateFullSearchSpace:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=False, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = matrices_combined_VAR, samples=samples_combined,alphas=[],couplingStrengths=couplingStrVAR_combined,noiseScales=noiseScales_combined,tauMaxs=[],
                                            runsPerConfig=runsPerConfigCombinedParams, fakeRun=fakeRun, startSeed= seed, evalType = "Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=False, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = matrices_combined_Cascade, samples=samples_combined,alphas=[],couplingStrengths=couplingStrCascade_combined,noiseScales=noiseScales_combined,tauMaxs=[],
                                            runsPerConfig=runsPerConfigCombinedParams, fakeRun=fakeRun, startSeed=seed, evalType = "Full")
            
            fullResults = np.stack((results_VAR, results_Cascade), axis=0)
            if doSave:
                np.save(folder + fullSearchFileName, fullResults)
        else:
            fullResults = np.load(folder+fullSearchFileName)
        fullResultsVAR = fullResults[0]
        fullResultsCascade = fullResults[1]

        #shape: usually 4 (TP/TN/FP/FN) or non-existant dim before, 3 (algorithms), runsPerConfig, len(couplingMatrices), len(samples), 1, len(couplingStrengths), len(noiseScales), 1
        fullResultsVAR = MCCFromFull(fullResultsVAR, axis=0)
        fullResultsCascade = MCCFromFull(fullResultsCascade, axis=0)

        # a data point has the following features: matrixSize, matrixDensity, samples, couplingStrength, noiseScale
        # with the label as some string indicating the ranking of algorithms

        feature_names = ["System Size", "System Density", "Samples", "Coupling Strength", "Noise Scale"]

        dataVAR=[]
        labelsVAR = []
        labelsMaxVAR = []
        for i in range(fullResultsVAR.shape[2]):
            for j in range(fullResultsVAR.shape[3]):
                for k in range(fullResultsVAR.shape[5]):
                    for m in range(fullResultsVAR.shape[6]):
                        for randomRun in range(fullResultsVAR.shape[1]):
                            dataVAR.append([int(i/3), i%3, samples_combined[j], couplingStrVAR_combined[k], noiseScales_combined[m]])
                            labelsVAR.append(getLabel(fullResultsVAR[:,randomRun, i,j,0,k,m,0]))
                            labelsMaxVAR.append(getMaxLabel(fullResultsVAR[:,randomRun, i,j,0,k,m,0]))
        labelsVAR = np.array(labelsVAR)
        labelsMaxVAR = np.array(labelsMaxVAR)
        dataVAR = np.array(dataVAR)
        class_names_max_VAR = ["GCSS", "LKIF", "PCMCI", "Equal"]
        class_names_max_VAR = class_names_max_VAR[np.min(labelsMaxVAR):]
        class_names_VAR = ["PCMCI-GCSS-LKIF", "GCSS-PCMCI-LKIF","GCSS-LKIF-PCMCI","PCMCI-LKIF-GCSS","LKIF-PCMCI-GCSS","LKIF-GCSS-PCMCI"]
        class_names_VAR = class_names_VAR[np.min(labelsVAR):]
        labelsVAR = labelsVAR - np.min(labelsVAR)
        labelsMaxVAR = labelsMaxVAR - np.min(labelsMaxVAR)
        for i in range(6):
            while i not in labelsVAR and i < np.max(labelsVAR):
                labelsVAR = labelsVAR-1
                del class_names_VAR[i]
        for i in range(4):
            while i not in labelsMaxVAR and i < np.max(labelsMaxVAR):
                labelsMaxVAR = labelsMaxVAR-1
                del class_names_max_VAR[i]

        stability = DataGenerator.getStabilityMatrix(matrices_combined_Cascade, samples_combined, couplingStrCascade_combined, noiseScales_combined, 1)
        dataCascade=[]
        labelsCascade = []
        labelsMaxCascade = []
        for i in range(fullResultsCascade.shape[2]):
            for j in range(fullResultsCascade.shape[3]):
                for k in range(fullResultsCascade.shape[5]):
                    for m in range(fullResultsCascade.shape[6]):
                        if stability[i,j,k,m] == 0:
                            for randomRun in range(fullResultsCascade.shape[1]):
                                dataCascade.append([int(i/3), i%3, samples_combined[j], couplingStrCascade_combined[k], noiseScales_combined[m]])
                                labelsCascade.append(getLabel(fullResultsCascade[:,randomRun, i,j,0,k,m,0]))
                                labelsMaxCascade.append(getMaxLabel(fullResultsCascade[:,randomRun, i,j,0,k,m,0]))
        labelsCascade = np.array(labelsCascade)
        labelsMaxCascade = np.array(labelsMaxCascade)
        dataCascade = np.array(dataCascade)
        class_names_max_Cascade = ["GCSS", "LKIF", "PCMCI", "Equal"]
        class_names_max_Cascade = class_names_max_Cascade[np.min(labelsMaxCascade):]
        class_names_Cascade = ["PCMCI-GCSS-LKIF", "GCSS-PCMCI-LKIF","GCSS-LKIF-PCMCI","PCMCI-LKIF-GCSS","LKIF-PCMCI-GCSS","LKIF-GCSS-PCMCI"]
        class_names_Cascade = class_names_Cascade[np.min(labelsCascade):]
        labelsCascade = labelsCascade - np.min(labelsCascade)
        for i in range(6):
            while i not in labelsCascade and i < np.max(labelsCascade):
                labelsCascade = labelsCascade-1
                del class_names_Cascade[i]
        for i in range(4):
            while i not in labelsMaxCascade and i < np.max(labelsMaxCascade):
                labelsMaxCascade = labelsMaxCascade-1
                del class_names_max_Cascade[i]

        vis.saveDecisionTree(dataVAR, labelsVAR, feature_names, class_names_VAR, folder + "VAR_DecTree.svg", show=showDiagrams, save= doSave)
        
        #vis.saveDecisionBoundaries(data = dataVAR[:,:2], labels=labelsVAR, filename = folder+ "VAR_MatrixDecBound", show= showDiagrams, save = doSave)

        vis.saveDecisionTree(dataCascade, labelsCascade, feature_names, class_names_Cascade, folder + "Cascade_DecTree.svg", show=showDiagrams, save= doSave)
        
        #vis.saveDecisionBoundaries(data = dataCascade[:,:2], labels=labelsCascade, filename = folder+ "Cascade_MatrixDecBound", show= showDiagrams, save = doSave)

        vis.saveDecisionTree(dataVAR, labelsMaxVAR, feature_names, class_names_max_VAR, folder + "VAR_DecTree_FirstRank.svg", show=showDiagrams, save= doSave)
        
        #vis.saveDecisionBoundaries(data = dataVAR[:,:2], labels=labelsMaxVAR, filename = folder+ "VAR_MatrixDecBound_FirstRank", show= showDiagrams, save = doSave)

        vis.saveDecisionTree(dataCascade, labelsMaxCascade, feature_names, class_names_max_Cascade, folder + "Cascade_DecTree_FirstRank.svg", show=showDiagrams, save= doSave)
        
        #vis.saveDecisionBoundaries(data = dataCascade[:,:2], labels=labelsMaxCascade, filename = folder+ "Cascade_MatrixDecBound_FirstRank", show= showDiagrams, save = doSave)

    if plotAlphaCombined:
        if calculateAlphaCombined:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=False, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=samples_combined,alphas=alphas_combined,couplingStrengths=couplingStrVAR_combined,noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaCombination, fakeRun=fakeRun, startSeed= seed, evalType = "Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=False, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=samples_combined,alphas=alphas_combined,couplingStrengths=couplingStrCascade_combined,noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaCombination, fakeRun=fakeRun, startSeed=seed, evalType = "Full")            
            fullResults = np.stack((results_VAR, results_Cascade), axis=0)
            if doSave:
                np.save(folder + alphaCombinedFileName, fullResults)
        else:
            fullResults = np.load(folder+alphaCombinedFileName)
        fullResultsVAR = fullResults[0]
        fullResultsCascade = fullResults[1]

        print(fullResultsCascade[:,1,:,0,3,0,3])
        print(fullResultsCascade[:,2,:,0,3,0,3])

        #shape: usually 4 (TP/TN/FP/FN) or non-existant dim before, 3 (algorithms), runsPerConfig, len(couplingMatrices), len(samples), len(alphas), len(couplingStrengths), len(noiseScales), len(tauMax)
        fullResultsVAR = MCCFromFull(fullResultsVAR, axis=0)
        fullResultsCascade = MCCFromFull(fullResultsCascade, axis=0)

        avgResultsVAR = np.average(fullResultsVAR, axis=1)
        avgResultsCasc = np.average(fullResultsCascade, axis=1)

        bestAlphaVAR = np.array(alphas_combined)[np.argmax(avgResultsVAR, axis=3).astype(int)].squeeze()
        bestAlphaCasc = np.array(alphas_combined)[np.argmax(avgResultsCasc, axis=3).astype(int)].squeeze()

        bestMCCVAR = np.max(avgResultsVAR, axis=3).squeeze()
        bestMCCCasc = np.max(avgResultsCasc, axis=3).squeeze()

        vis.saveHeatmap(bestAlphaVAR[0].T, bestMCCVAR[0].T, "", filename = folder + "GCSS_alphaComb_VAR", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrVAR_combined)
        vis.saveHeatmap(bestAlphaVAR[1].T, bestMCCVAR[1].T, "", filename = folder + "LKIF_alphaComb_VAR", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrVAR_combined)
        vis.saveHeatmap(bestAlphaVAR[2].T, bestMCCVAR[2].T, "", filename = folder + "PCMCI_alphaComb_VAR", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrVAR_combined)
        
        vis.saveHeatmap(bestAlphaCasc[0].T, bestMCCCasc[0].T, "", filename = folder + "GCSS_alphaComb_Cascade", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrCascade_combined)
        vis.saveHeatmap(bestAlphaCasc[1].T, bestMCCCasc[1].T, "", filename = folder + "LKIF_alphaComb_Cascade", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrCascade_combined)
        vis.saveHeatmap(bestAlphaCasc[2].T, bestMCCCasc[2].T, "", filename = folder + "PCMCI_alphaComb_Cascade", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrCascade_combined)

    if plotBonusAlpha:
        if calculateBonusAlpha:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaBonus, fakeRun=fakeRun, startSeed= seed,iterateSeeds=False, evalType="Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaBonus, fakeRun=fakeRun, startSeed=seed,iterateSeeds=False, evalType="Full")
            results_VAR = results_VAR[2]
            results_Cascade = results_Cascade[2]
            resultsAlpha = np.stack((results_VAR, results_Cascade), axis=0)
            if doSave:
                np.save(folder + "bonusAlpha.npy", resultsAlpha)
        else:
            resultsAlpha = np.load(folder+"bonusAlpha.npy")
        if calculateBonusAlpha2:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaBonus, fakeRun=fakeRun, startSeed= seed,iterateSeeds=True, evalType="Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaBonus, fakeRun=fakeRun, startSeed=seed,iterateSeeds=True, evalType="Full")
            results_VAR = results_VAR[2]
            results_Cascade = results_Cascade[2]
            resultsAlpha2 = np.stack((results_VAR, results_Cascade), axis=0)
            if doSave:
                np.save(folder + "bonusAlpha2.npy", resultsAlpha2)
        else:
            resultsAlpha2 = np.load(folder + "bonusAlpha2.npy")

        print(np.squeeze(resultsAlpha[0,:,2,:,10]))
        print(np.squeeze(resultsAlpha2[0,:,2,:,10]))

        resultsAlpha = np.average(tpr_fpr_FromFull(resultsAlpha, axis= 1), axis= 3)
        # visualization for default configuration with 3 vertices, 4 edges
        vis.saveROCCurve(resultsAlpha[:,0,0,:], resultsAlpha[:,1,0,:], alphas, "GCSS ROC Curve for alpha values", folder+"GCSS_Alpha_VAR_Cascade_bonus", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsAlpha[:,0,1,:], resultsAlpha[:,1,1,:], alphas, "LKIF ROC Curve for alpha values", folder+"LKIF_Alpha_VAR_Cascade_bonus", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsAlpha[:,0,2,:], resultsAlpha[:,1,2,:], alphas, "PCMCI ROC Curve for alpha values", folder+"PCMCI_Alpha_VAR_Cascade_bonus", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        
        resultsAlpha2 = np.average(tpr_fpr_FromFull(resultsAlpha2, axis= 1), axis= 3)
        # visualization for default configuration with 3 vertices, 4 edges
        vis.saveROCCurve(resultsAlpha2[:,0,0,:], resultsAlpha2[:,1,0,:], alphas, "GCSS ROC Curve for alpha values", folder+"GCSS_Alpha_VAR_Cascade", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsAlpha2[:,0,1,:], resultsAlpha2[:,1,1,:], alphas, "LKIF ROC Curve for alpha values", folder+"LKIF_Alpha_VAR_Cascade", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsAlpha2[:,0,2,:], resultsAlpha2[:,1,2,:], alphas, "PCMCI ROC Curve for alpha values", folder+"PCMCI_Alpha_VAR_Cascade", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")


    endTime = time.process_time()
    print("Total execution time in seconds: ")
    print(endTime - startTime)
    print(str(int((endTime - startTime)/60))+ " minutes")
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-testIndex", type=int, choices=[0,1,2,3,4], help="0 to run all analysis, 1 for alpha, 2 for tau-max, 3 for separate data parameters, 4 for combined data parameters, 5 for a reduced combination of alpha values, sampling and coupling strength", required=False)
    args = parser.parse_args()
    mainEvaluations(args.testIndex)