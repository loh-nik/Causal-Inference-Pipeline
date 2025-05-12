import numpy as np
import DataGenerator
import GCSS
import LKIF
import PCMCI
import Visualization as vis
#from matplotlib import pyplot as plt
from progress.bar import Bar
import time

def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

def maxSignificantLink(x, bool_matr, axis = None):
    y = np.multiply(x, bool_matr)
    return absmaxND(y, axis = axis)

def getMeanStdDev(data, axis = None):
    return np.average(data, axis = axis), np.std(data, axis = axis)

folder = "./GridExperiment/"

globalVarTerm = "Global"

# generate a single grid at default settings, no coupling, with different alpha values
def standaloneGrid(show = False, save = True):
    seed = 0

    gridSize = 10
    samples = 500
    slowness = 0.5
    alphas = [0.01, 0.05, 0.5]
    cellNoise = 0.1
    couplingFactor = 0.25 * (1 / np.power(2,slowness))
    driver, grid = DataGenerator.getGridData(gridX = gridSize, gridY = gridSize, continuousGrid=True, couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2),
                                       borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, startVal=0, autoCoupling = 1 - (couplingFactor*4), driverCoupling = 0, 
                                       gridToDriverCoupling=0, gridToDriverDelay=1, selfNoise= cellNoise, driverAutoReg=0, driverNoise = 0, timeSteps=samples, seed = seed)
    for alpha in alphas:
        matrix = LKIF.lkif(grid, alpha)
        np.fill_diagonal(matrix, 0)
        vis.saveGrid(matrix, "Detected Coupling Matrix\nError Rate " + str(alpha), folder + "GridAlpha_"+str(alpha)+".png", show=show, save = save)

# grid with interactions
def gridInteractionsSpatialDist(show=False, save = True):
    gridSize = 10
    seed = 0
    samples = 500
    slowness = 0.5
    cellNoises = [0.1, 0.5, 2]
    couplingFactor = 0.25 * (1 / np.power(2,slowness))
    for cellNoise in cellNoises:
        # driver to grid only
        driver, grid = DataGenerator.getGridData(gridX = gridSize, gridY = gridSize, continuousGrid=True, couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2),
                                       borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, startVal=0, autoCoupling = 1 - (couplingFactor*4), driverCoupling = 1, 
                                       gridToDriverCoupling=0, gridToDriverDelay=1, selfNoise= cellNoise, driverAutoReg=0.5, driverNoise = 0.5, timeSteps=samples, seed = seed)
        pairwiseMatrix = np.zeros((2, gridSize*gridSize))
        for i in range(gridSize*gridSize):
            localTest = np.append(grid[i].reshape(1,samples), driver.reshape(1,samples),axis = 0)
            localMat = LKIF.lkif(localTest, 0.05)
            pairwiseMatrix[0,i] = localMat[1,0]
            pairwiseMatrix[1,i] = localMat[0,1]
        pairwiseResult = np.reshape(pairwiseMatrix[0], (gridSize,-1))
        vis.saveGrid(pairwiseResult, "Pairwise - " + globalVarTerm + " to Grid Cells\nCell Noise Level "+ str(cellNoise), folder + "DriverToGrid_Pairwise_"+str(cellNoise)+".png", show=show, save=save)
        
        grid2 = np.append(grid, driver.reshape(1,samples), axis=0)
        matrix = LKIF.lkif(grid2, 0.05)
        np.fill_diagonal(matrix, 0)
        rowFrom = np.reshape(matrix[gridSize*gridSize, :gridSize*gridSize],(gridSize,-1))
        vis.saveGrid(rowFrom, "Full Analysis - " + globalVarTerm + " to Grid Cells\nCell Noise Level "+ str(cellNoise), folder + "DriverToGrid_"+str(cellNoise)+".png", show=show, save=save)

        # grid to driver only
        driver, grid = DataGenerator.getGridData(gridX = gridSize, gridY = gridSize, continuousGrid=True, couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2),
                                       borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, startVal=0, autoCoupling = 1 - (couplingFactor*4), driverCoupling = 0, 
                                       gridToDriverCoupling=1 / (gridSize*gridSize), gridToDriverDelay=1, selfNoise= cellNoise, driverAutoReg=0.5, driverNoise = 0.5, timeSteps=samples, seed = seed)
        pairwiseMatrix = np.zeros((2, gridSize*gridSize))
        for i in range(gridSize*gridSize):
            localTest = np.append(grid[i].reshape(1,samples), driver.reshape(1,samples),axis = 0)
            localMat = LKIF.lkif(localTest, 0.05)
            pairwiseMatrix[0,i] = localMat[1,0]
            pairwiseMatrix[1,i] = localMat[0,1]
        pairwiseResult = np.reshape(pairwiseMatrix[1], (gridSize,-1))
        vis.saveGrid(pairwiseResult, "Pairwise - Grid Cells to " + globalVarTerm + "\nCell Noise Level "+ str(cellNoise), folder + "GridToDriver_Pairwise_"+str(cellNoise)+".png", show=show, save=save)
        
        grid2 = np.append(grid, driver.reshape(1,samples), axis=0)
        matrix = LKIF.lkif(grid2, 0.05)
        np.fill_diagonal(matrix, 0)
        rowTo = np.reshape(matrix[:gridSize*gridSize,gridSize*gridSize],(gridSize,-1))
        vis.saveGrid(rowTo, "Full Analysis - Grid Cells to " + globalVarTerm + "\nCell Noise Level "+ str(cellNoise), folder + "GridToDriver_"+str(cellNoise)+".png", show=show, save=save)

# grid with interactions
def gridInteractionsUnidir(show=False, save = True, calculate = True):
    # experiment with all three algorithms on aggregates of cells, with varying grid size
    slownesses = [0,0.25,0.5,1,2,3,5]
    cellNoise = 0.5
    runs = 10
    samples = 500
    gridSize = 10
    if calculate:
        # 3 algorithms
        absAggrEffDrivToGrid = np.zeros((len(slownesses),runs,3))
        absAggrEffGridToDriv = np.zeros((len(slownesses),runs,3))
        absAggEffNoCausality_DrivToGrid = np.zeros((len(slownesses),runs,3))
        absAggEffNoCausality_GridToDriv = np.zeros((len(slownesses),runs,3))
        progressBar = Bar('Processing', max=runs*len(slownesses))
        for slownessIndex in range(len(slownesses)):
            slowness = slownesses[slownessIndex]
            couplingFactor = 0.25 * (1 / np.power(2,slowness))
            for run in range(runs):
                # driver to grid only
                driver, grid = DataGenerator.getGridData(gridX = gridSize, gridY = gridSize, continuousGrid=True, couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2),
                                            borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, startVal=0, autoCoupling = 1 - (couplingFactor*4), driverCoupling = 1, 
                                            gridToDriverCoupling=0, gridToDriverDelay=1, selfNoise= cellNoise, driverAutoReg=0.5, driverNoise = 0.5, timeSteps=samples, seed = run)
                gridAggr = np.average(grid, axis=0)
                grid2 = np.append(gridAggr.reshape(1,samples), driver.reshape(1,samples),axis = 0)
                LKIFResult = LKIF.lkif(grid2, 0.05)[1,0]
                matrix, p_values = PCMCI.PCMCIPlus(grid2.T, [], [0,1], None, contempLinks=False)
                graph_bool = p_values <= 0.05
                PCMCIResult = maxSignificantLink(matrix, graph_bool, axis = 2)[1,0]
                try:
                    GCSSResult = GCSS.gcss(grid2, 0.05, 5, returnAll=False).T[1,0]
                except:
                    GCSSResult = np.zeros((grid2.shape[0],grid2.shape[0]))
                    print("Error on GCSS")
                    exit()
                absAggrEffDrivToGrid[slownessIndex, run,:] = [LKIFResult, PCMCIResult, GCSSResult]
                

                # grid to driver only
                driver, grid = DataGenerator.getGridData(gridX = gridSize, gridY = gridSize, continuousGrid=True, couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2),
                                            borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, startVal=0, autoCoupling = 1 - (couplingFactor*4), driverCoupling = 0, 
                                            gridToDriverCoupling=5 / (gridSize*gridSize), gridToDriverDelay=1, selfNoise= cellNoise, driverAutoReg=0.5, driverNoise = 0.5, timeSteps=samples, seed = run)
                gridAggr = np.average(grid, axis=0)
                grid2 = np.append(gridAggr.reshape(1,samples), driver.reshape(1,samples),axis = 0)
                LKIFResult = LKIF.lkif(grid2, 0.05)[0,1]
                matrix, p_values = PCMCI.PCMCIPlus(grid2.T, [], [0,1], None, contempLinks=False)
                graph_bool = p_values <= 0.05
                PCMCIResult = maxSignificantLink(matrix, graph_bool, axis = 2)[0,1]
                try:
                    GCSSResult = GCSS.gcss(grid2, 0.05, 5, returnAll=False).T[0,1]
                except:
                    GCSSResult = np.zeros((grid2.shape[0],grid2.shape[0]))
                    print("Error on GCSS")
                    exit()
                absAggrEffGridToDriv[slownessIndex, run,:] = [LKIFResult, PCMCIResult, GCSSResult]

                # zero causality
                driver, grid = DataGenerator.getGridData(gridX = gridSize, gridY = gridSize, continuousGrid=True, couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2),
                                            borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, startVal=0, autoCoupling = 1 - (couplingFactor*4), driverCoupling = 0, 
                                            gridToDriverCoupling= 0, gridToDriverDelay=1, selfNoise= cellNoise, driverAutoReg=0.5, driverNoise = 0.5, timeSteps=samples, seed = run)
                gridAggr = np.average(grid, axis=0)
                grid2 = np.append(gridAggr.reshape(1,samples), driver.reshape(1,samples),axis = 0)
                LKIFResult = LKIF.lkif(grid2, 0.05)
                matrix, p_values = PCMCI.PCMCIPlus(grid2.T, [], [0,1], None, contempLinks=False)
                graph_bool = p_values <= 0.05
                PCMCIResult = maxSignificantLink(matrix, graph_bool, axis = 2)
                try:
                    GCSSResult = GCSS.gcss(grid2, 0.05, 5, returnAll=False).T
                except:
                    GCSSResult = np.zeros((grid2.shape[0],grid2.shape[0]))
                    print("Error on GCSS")
                    exit()
                absAggEffNoCausality_DrivToGrid[slownessIndex, run,:] = [LKIFResult[1,0], PCMCIResult[1,0], GCSSResult[1,0]]
                absAggEffNoCausality_GridToDriv[slownessIndex, run, :]= [LKIFResult[0,1], PCMCIResult[0,1], GCSSResult[0,1]]
                progressBar.next()
        progressBar.finish()
        # we want shape (4,gridSizeParams) for each algorithm
        stacked = np.stack((absAggrEffDrivToGrid, absAggrEffGridToDriv, absAggEffNoCausality_DrivToGrid, absAggEffNoCausality_GridToDriv), axis= 0)
        if save:
            np.save("unidirectional_Slowness_GridAggregates.npy", stacked)
    else:
        stacked = np.load("unidirectional_Slowness_GridAggregates.npy")
    stackedAvg, stackedStd = getMeanStdDev(stacked, axis=2)
    vis.saveF1Curve(stackedAvg[:,:,0], slownesses, "LKIF - Causal Effects between " + globalVarTerm + "\nand Grid for Unidirectional Effects", folder + "Unidir_Coupling_Slowness_LKIF", 
                    stackedStd[:,:,0], rowLabels = ["" + globalVarTerm + " to Grid", "Grid to " + globalVarTerm + "", "" + globalVarTerm + " to Grid FP", "Grid to " + globalVarTerm + " FP"],
                    show = show, save = save, figsize = (4.5,4),dpi=300, xlabel ="Diffusion Factor", ylabel = "Causal Effect Strength")
    vis.saveF1Curve(stackedAvg[:,:,1], slownesses, "PCMCI - Causal Effects between " + globalVarTerm + "\nand Grid for Unidirectional Effects", folder + "Unidir_Coupling_Slowness_PCMCI", 
                    stackedStd[:,:,1], rowLabels = ["" + globalVarTerm + " to Grid", "Grid to " + globalVarTerm + "", "" + globalVarTerm + " to Grid FP", "Grid to " + globalVarTerm + " FP"],
                    show = show, save = save, figsize = (4.5,4),dpi=300, xlabel ="Diffusion Factor", ylabel = "Causal Effect Strength")
    vis.saveF1Curve(stackedAvg[:,:,2], slownesses, "GCSS - Causal Effects between " + globalVarTerm + "\nand Grid for Unidirectional Effects", folder + "Unidir_Coupling_Slowness_GCSS", 
                    stackedStd[:,:,2], rowLabels = ["" + globalVarTerm + " to Grid", "Grid to " + globalVarTerm + "", "" + globalVarTerm + " to Grid FP", "Grid to " + globalVarTerm + " FP"],
                    show = show, save = save, figsize = (4.5,4),dpi=300, xlabel ="Diffusion Factor", ylabel = "Causal Effect Strength")

# grid with interactions
def gridInteractionsBidir(show=False, save = True, calculate = True):
    # fraction of grid coupling constant over driv coupling constant
    fracGridDriv = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    cellNoise = 0.5
    runs = 10
    samples = 500
    gridSize = 10
    # in thesis with 0.5, for appendix with 4
    slowness = 4
    couplingFactor = 0.25 * (1 / np.power(2,slowness))
    if calculate:
        # 3 algorithms
        absAggrEffDrivToGrid = np.zeros((len(fracGridDriv),runs,3))
        absAggrEffGridToDriv = np.zeros((len(fracGridDriv),runs,3))
        absAggEffNoCausality_DrivToGrid = np.zeros((len(fracGridDriv),runs,3))
        absAggEffNoCausality_GridToDriv = np.zeros((len(fracGridDriv),runs,3))

        absAggEff_loggingLKIF = np.zeros((len(fracGridDriv),runs,3))
        progressBar = Bar('Processing', max=runs*len(fracGridDriv))
        for fractionIndex in range(len(fracGridDriv)):
            frac = fracGridDriv[fractionIndex]
            for run in range(runs):
                # bidirectional
                driver, grid = DataGenerator.getGridData(gridX = gridSize, gridY = gridSize, continuousGrid=True, couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2),
                                            borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, startVal=0, autoCoupling = 1 - (couplingFactor*4), driverCoupling = 0.2/frac, 
                                            gridToDriverCoupling= 0.5*frac / (gridSize*gridSize), gridToDriverDelay=1, selfNoise= cellNoise, driverAutoReg=0.5, driverNoise = 0.5, timeSteps=samples, seed = run)
                gridAggr = np.average(grid, axis=0)
                grid2 = np.append(gridAggr.reshape(1,samples), driver.reshape(1,samples),axis = 0)
                LKIFResult = LKIF.lkif(grid2, 0.05)
                matrix, p_values = PCMCI.PCMCIPlus(grid2.T, [], [0,1], None, contempLinks=False)
                graph_bool = p_values <= 0.05
                PCMCIResult = maxSignificantLink(matrix, graph_bool, axis = 2)
                try:
                    GCSSResult = GCSS.gcss(grid2, 0.05, 5, returnAll=False).T
                except:
                    GCSSResult = np.zeros((grid2.shape[0],grid2.shape[0]))
                    print("Error on GCSS")
                    exit()
                absAggrEffDrivToGrid[fractionIndex, run,:] = [LKIFResult[1,0], PCMCIResult[1,0], GCSSResult[1,0]]
                absAggrEffGridToDriv[fractionIndex, run, :] =  [LKIFResult[0,1], PCMCIResult[0,1], GCSSResult[0,1]]
                
                # grid to driver only
                driver, grid = DataGenerator.getGridData(gridX = gridSize, gridY = gridSize, continuousGrid=True, couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2),
                                            borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, startVal=0, autoCoupling = 1 - (couplingFactor*4), driverCoupling = 0, 
                                            gridToDriverCoupling= 0.5*frac / (gridSize*gridSize), gridToDriverDelay=1, selfNoise= cellNoise, driverAutoReg=0.5, driverNoise = 0.5, timeSteps=samples, seed = run)
                gridAggr = np.average(grid, axis=0)
                grid2 = np.append(gridAggr.reshape(1,samples), driver.reshape(1,samples),axis = 0)
                LKIFResult = LKIF.lkif(grid2, 0.05)
                matrix, p_values = PCMCI.PCMCIPlus(grid2.T, [], [0,1], None, contempLinks=False)
                graph_bool = p_values <= 0.05
                PCMCIResult = maxSignificantLink(matrix, graph_bool, axis = 2)
                try:
                    GCSSResult = GCSS.gcss(grid2, 0.05, 5, returnAll=False).T
                except:
                    GCSSResult = np.zeros((grid2.shape[0],grid2.shape[0]))
                    print("Error on GCSS")
                    exit()
                absAggEffNoCausality_DrivToGrid[fractionIndex, run,:] = [LKIFResult[1,0], PCMCIResult[1,0], GCSSResult[1,0]]

                absAggEff_loggingLKIF[fractionIndex, run,:] = [LKIFResult[0,1], PCMCIResult[0,1], GCSSResult[0,1]]
                
                # driver to grid only
                driver, grid = DataGenerator.getGridData(gridX = gridSize, gridY = gridSize, continuousGrid=True, couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2),
                                            borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, startVal=0, autoCoupling = 1 - (couplingFactor*4), driverCoupling = 0.2 / frac, 
                                            gridToDriverCoupling= 0, gridToDriverDelay=1, selfNoise= cellNoise, driverAutoReg=0.5, driverNoise = 0.5, timeSteps=samples, seed = run)
                gridAggr = np.average(grid, axis=0)
                grid2 = np.append(gridAggr.reshape(1,samples), driver.reshape(1,samples),axis = 0)
                LKIFResult = LKIF.lkif(grid2, 0.05)
                matrix, p_values = PCMCI.PCMCIPlus(grid2.T, [], [0,1], None, contempLinks=False)
                graph_bool = p_values <= 0.05
                PCMCIResult = maxSignificantLink(matrix, graph_bool, axis = 2)
                try:
                    GCSSResult = GCSS.gcss(grid2, 0.05, 5, returnAll=False).T
                except:
                    GCSSResult = np.zeros((grid2.shape[0],grid2.shape[0]))
                    print("Error on GCSS")
                    exit()
                absAggEffNoCausality_GridToDriv[fractionIndex, run, :]= [LKIFResult[0,1], PCMCIResult[0,1], GCSSResult[0,1]]
                progressBar.next()
        progressBar.finish()
        print("Causality grid driv bidir.: ")
        print(absAggrEffDrivToGrid[6,:,0])
        print("Causality grid driv unidir.: ")
        print(absAggEff_loggingLKIF[6,:,0])
        # we want shape (4,gridSizeParams) for each algorithm
        stacked = np.stack((absAggrEffDrivToGrid, absAggrEffGridToDriv, absAggEffNoCausality_DrivToGrid, absAggEffNoCausality_GridToDriv), axis= 0)
        if save:
            np.save("bidirectional_GridAggregates_Slowness4.npy", stacked)
    else:
        stacked = np.load("bidirectional_GridAggregates.npy")
    stackedAvg, stackedStd = getMeanStdDev(stacked, axis=2)
    vis.saveF1Curve(stackedAvg[:,:,0], fracGridDriv, "LKIF - Causal Effects between " + globalVarTerm + " \nand Grid for Bidirectional Effects", folder + "Bidir_Coupling_LKIF", 
                    stackedStd[:,:,0], rowLabels = ["" + globalVarTerm + " to Grid", "Grid to " + globalVarTerm + "", "" + globalVarTerm + " to Grid FP", "Grid to " + globalVarTerm + " FP"],
                    show = show, save = save, figsize = (4.5,4.5),dpi=300, xlabel ="Coupling Strength of Grid -> " + globalVarTerm + "\ndiv. by " + globalVarTerm + " -> Grid", ylabel = "Causal Effect Strength", xscale = "log")
    vis.saveF1Curve(stackedAvg[:,:,1], fracGridDriv, "PCMCI - Causal Effects between " + globalVarTerm + " \nand Grid for Bidirectional Effects", folder + "Bidir_Coupling_PCMCI", 
                    stackedStd[:,:,1], rowLabels = ["" + globalVarTerm + " to Grid", "Grid to " + globalVarTerm + "", "" + globalVarTerm + " to Grid FP", "Grid to " + globalVarTerm + " FP"],
                    show = show, save = save, figsize = (4.5,4.5),dpi=300, xlabel ="Coupling Strength of Grid -> " + globalVarTerm + "\ndiv. by " + globalVarTerm + " -> Grid", ylabel = "Causal Effect Strength", xscale = "log")
    vis.saveF1Curve(stackedAvg[:,:,2], fracGridDriv, "GCSS - Causal Effects between " + globalVarTerm + " \nand Grid for Bidirectional Effects", folder + "Bidir_Coupling_GCSS", 
                    stackedStd[:,:,2], rowLabels = ["" + globalVarTerm + " to Grid", "Grid to " + globalVarTerm + "", "" + globalVarTerm + " to Grid FP", "Grid to " + globalVarTerm + " FP"],
                    show = show, save = save, figsize = (4.5,4.5),dpi=300, xlabel ="Coupling Strength of Grid -> " + globalVarTerm + "\ndiv. by " + globalVarTerm + " -> Grid", ylabel = "Causal Effect Strength", xscale = "log", yAxisCut=True, yAxisLinearLim=0.5)

if __name__ == "__main__":
    #standaloneGrid(show=False, save=True)
    #gridInteractionsSpatialDist(show=False, save = True)
    #gridInteractionsUnidir(show=False, save=True, calculate=False)
    gridInteractionsBidir(show=False, save=True, calculate=False)