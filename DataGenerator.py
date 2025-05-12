import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from tigramite.toymodels import structural_causal_processes as toys
from progress.bar import Bar

def dynSysX3(t,y,A,B,C,D,deltaT,noiseArr):
    return A*(np.power(y,3)) + B*y + C + D@(y-1) + noiseArr[int(t/deltaT)%len(noiseArr)]

def getCascadeData(couplingMatrix, samples, deltaTOutput, noiseScale, seed, constantNoiseDuration = 1, deltaTSim = 0.001):
    n, _ = couplingMatrix.shape
    endT = samples*deltaTOutput
    timeSpan = [0, endT]
    timePoints = [x * deltaTSim for x in range(int(endT * (1/deltaTSim)))]
    randomGen = np.random.default_rng(seed)
    noiseArray = randomGen.normal(loc=0,scale=noiseScale,size=(int((1/constantNoiseDuration)*endT / deltaTOutput), n))
    init = np.ones(n)
    A = np.ones(n) * (-1)
    B = np.ones(n)
    C = np.zeros(n)
    sol = solve_ivp(dynSysX3, timeSpan, init, t_eval = timePoints, args=(A,B,C,couplingMatrix,deltaTOutput*constantNoiseDuration,noiseArray))
    return sol.y.T[::int(deltaTOutput/deltaTSim)]

def getVARData(couplingMatrix, samples, noiseScale, seed):
    n,_ = couplingMatrix.shape
    randomGen = np.random.default_rng(seed)
    noiseArray = randomGen.normal(loc = 0, scale = noiseScale, size=(int(1.1*samples), n))
    links = {}
    def lin(x): return x
    for i in range(n):
        row = []
        for j in range(n):
            if couplingMatrix[i,j] != 0:
                row.append(((j, -1), couplingMatrix[i,j], lin))
        links[i] = row
    data_full, nonstationarity_indicator = toys.structural_causal_process(
        links=links, T=samples, noises=noiseArray, seed=seed, transient_fraction=0.1)
    assert not nonstationarity_indicator
    return data_full

# corner and border treatment are a bit confusing - for "energy conservation" (or stability around startVal), one needs to use the following formula when calling:
# couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2), borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, autoCoupling = 1 - (couplingFactor*4), startVal=XY
def getGridData(gridX, gridY, couplingGrid, autoCoupling, driverCoupling, selfNoise, timeSteps, continuousGrid = False, 
                startVal = 0, cornerAuto = 0.5, borderAuto = 0.25, borderLoss = -0.25, gridToDriverCoupling=0, gridToDriverDelay=4, 
                driverAutoReg = 0, driverNoise=1, driverTimeSeries = [], driverNoiseSeries = [], seed = 0):
    randomGen = np.random.default_rng(seed)
    valuesGrid = startVal * np.ones((gridX * gridY, timeSteps))
    # to allow for energy dissipation to the border of the grid, basically modelling the grid to be surrounded by cells with constant value startVal
    constLoss = np.zeros((gridX * gridY))
    lossValue = - borderLoss * startVal
    generateDriver = False
    generateNoise = False
    if driverTimeSeries == []:
        driverTimeSeries = np.zeros(timeSteps)
        if driverNoiseSeries == []:
            driverNoiseSeries = np.zeros(timeSteps)
            generateNoise = True
        generateDriver = True
        
    couplingMatrix = np.zeros((gridX*gridY, gridX * gridY))
    # could speed this up by writing more elegantly, whatever
    if continuousGrid:
        for i in range(gridX*gridY):
            couplingMatrix[i, i-gridX] = couplingGrid
            couplingMatrix[i,(i+gridX)%(gridX*gridY)] = couplingGrid
            couplingMatrix[i,i-1] = couplingGrid
            couplingMatrix[i,(i+1)%(gridX*gridY)] = couplingGrid
    else:
        for i in range(gridX*gridY):
            if i == 0 or i == gridX-1 or i == gridX * gridY - 1 or i == gridX * (gridY-1):
                couplingMatrix[i,i] = cornerAuto + (2 * borderLoss)
                constLoss[i] = lossValue * 2
            elif i < gridX or i % gridX == 0 or i%gridX == gridX-1 or i > gridX * (gridY-1):
                couplingMatrix[i,i] = borderAuto + borderLoss
                constLoss[i] = lossValue
            else:
                couplingMatrix[i,i] = autoCoupling
            if i >= gridX:
                couplingMatrix[i, i-gridX] = couplingGrid
            if i < (gridX*gridY)-gridX:
                couplingMatrix[i,i+gridX] = couplingGrid
            if i % gridX > 0:
                couplingMatrix[i,i-1] = couplingGrid
            if i % gridX != (gridX-1):
                couplingMatrix[i,i+1] = couplingGrid
    driverCoupling = np.ones(gridX*gridY) * driverCoupling
    for j in range(1,timeSteps):
        valuesGrid[:,j] = couplingMatrix@valuesGrid[:,j-1] + constLoss + driverCoupling * (driverTimeSeries[j-1] - startVal) + randomGen.normal(loc = 0, scale = selfNoise, size=gridX*gridY)
        if generateDriver:
            driverTimeSeries[j] = driverAutoReg * driverTimeSeries[j-1] + driverNoiseSeries[j] + (randomGen.normal(loc=0,scale=driverNoise) if generateNoise else 0) + (0 if j <= (gridToDriverDelay-1) else np.sum(gridToDriverCoupling * valuesGrid[:,j-gridToDriverDelay]))
    if np.max(driverTimeSeries) > 100 or np.max(valuesGrid) > 100 or np.any(np.isnan(driverTimeSeries)) or np.any(np.isnan(valuesGrid)):
        print("error: values larger than 100 indicate lack of convergence / parametrization fault")
        exit()
    return driverTimeSeries, valuesGrid

# returns a multi-dim array that contains 1 for instability and 0 for stability in cascade data gen
def getStabilityMatrix(couplingMatrices, sampleCounts, couplingStrengths, noiseScales, runs):
    result = np.zeros((len(couplingMatrices), len(sampleCounts), len(couplingStrengths), len(noiseScales)))
    totalRuns = (len(couplingMatrices) * len(sampleCounts) * len(couplingStrengths) * len(noiseScales)) * runs
    progressBar = Bar('Processing', max=totalRuns)
    for matInd in range(len(couplingMatrices)):
        for sampInd in range(len(sampleCounts)):
            for strength in range(len(couplingStrengths)):
                matrix = couplingStrengths[strength] * couplingMatrices[matInd]
                for i in range(matrix.shape[0]):
                    matrix[i,i] = couplingMatrices[matInd][i,i]
                for noiseInd in range(len(noiseScales)):
                    for run in range(runs):
                        progressBar.next()
                        data = getCascadeData(couplingMatrices[matInd], sampleCounts[sampInd], 0.1, noiseScales[noiseInd], run, deltaTSim=0.01)
                        if np.min(data) <= -1:
                            result[matInd, sampInd, strength, noiseInd] = 1
                            print("Unstable config for matrix " + str(matInd) + ", " + str(sampleCounts[sampInd]) + " samples, coupling strength " + str(couplingStrengths[strength]) + " and noise scale " + str(noiseScales[noiseInd]))
    progressBar.finish()
    return result

def checkStability():
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
    highDensityCouplingMatrixVAR = np.array([
                                        [0.5,0,-1,-1,0,0],
                                        [1,0.5,0,0,0,0],
                                        [1,1,0.5,0,0,0],
                                        [0,0,0,0.5,-1,-1],
                                        [0,1,-1,0,0.5,1],
                                        [0,0,0,1,-1,0.5]])
    highDensityCouplingMatrixCascade = np.array([
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
    defaultCouplingMatrixCascade = np.array([[0,1,0],[-1,0,0],[1,-1,0]])
    for i in [mediumCouplingMatrixVAR, largeCouplingMatrixVAR, highDensityCouplingMatrixVAR, defaultCouplingMatrixVAR_LowDense, defaultCouplingMatrixVAR_HighDense, largeCouplingMatrixVAR_LowDense,largeCouplingMatrixVAR_HighDense]:
        matrix = 0.1 * i
        for m in range(matrix.shape[0]):
            matrix[m,m] = i[m,m]
        #data = getVARData(matrix, 2000, 0.1, 0)
        #plt.figure()
        #for j in range(data.shape[1]):
        #    plt.plot(data[:,j])
    #for i in [mediumCouplingMatrixCascade, largeCouplingMatrixCascade, highDensityCouplingMatrixCascade,defaultCouplingMatrixCascade_LowDense,defaultCouplingMatrixCascade_HighDense,largeCouplingMatrixCascade_LowDense,largeCouplingMatrixCascade_HighDense ]:
    for i in [defaultCouplingMatrixCascade]: 
        data = getCascadeData(3*i, 10000, 0.1,0.1, 0)
        plt.figure()
        for j in range(data.shape[1]):
            plt.plot(data[:,j], label = str(j))
        plt.legend()
    plt.show()


if __name__ == "__main__":
    checkStability()