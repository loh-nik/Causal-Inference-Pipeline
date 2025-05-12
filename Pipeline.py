import argparse
import os
import sys
import copy
import LKIF
import GCSS
from PCMCI import PCMCIPlus
from scipy.signal import detrend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.toymodels import structural_causal_processes as toys
from tigramite.causal_effects import CausalEffects
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite import plotting as tp
from tigramite.models import LinearMediation
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

# data of shape (variable, time steps)
def preprocessData(data, detrending = True, deseason = True, diff = True, deseasonLength = 12):
    d,n = data.shape
    currentData = copy.deepcopy(data)
    censor = np.isnan(currentData)
    currentData[censor] = 0
    if detrending:
        currentData = detrend(currentData)
    if deseason:
        if n%deseasonLength > 0:
            avg = np.average(np.reshape(currentData[:,0:(n - (n % deseasonLength))], (d,deseasonLength,-1), 'F'), axis=2).reshape(d,deseasonLength,1)
            for i in range(deseasonLength):
                currentData[:,i:len(currentData):deseasonLength] = currentData[:,i:len(currentData):deseasonLength] - avg[:,i]
        else:
            avg = np.average(np.reshape(currentData, (d,deseasonLength,-1), 'F'), axis=2)
            deseason = np.subtract(np.reshape(currentData, (d,deseasonLength,-1), 'F'), avg.reshape(d,deseasonLength,1))
            deseason = np.reshape(deseason, (d,-1), order='F')
            currentData = deseason
    if diff:
        temp = np.zeros(currentData.shape)
        for i in range(n-1):
            temp[:,i+1] = currentData[:,i+1] - currentData[:,i]
        currentData = temp
    currentData[censor] = np.nan
    return currentData

def visualizeGraph(val_matrix, graph, columnNames, filename, title="", show = False, save = True):
    plotMatrix(val_matrix, columnNames, filename + "matrix")
    fig = plt.subplots(1,1,layout="constrained")
    plt.suptitle(title, fontsize=14)
    if val_matrix.shape[0] == 2:
        figur, ax = fig
        figur.set_size_inches(4,2)
        if len(graph) > 0:
            tp.plot_graph(
            val_matrix=val_matrix,
            graph=graph,
            var_names=columnNames,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            show_autodependency_lags=False,
            node_size = 0.4,
            node_aspect = 2,
            node_label_size=12,
            link_label_fontsize=12,
            fig_ax=fig
            )
        else:
            matrixFull = np.zeros((val_matrix.shape[0], val_matrix.shape[1],2))
            matrixFull[:,:,1] = val_matrix
            tp.plot_graph(
            val_matrix=matrixFull,
            graph=matrixFull,
            var_names=columnNames,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            show_autodependency_lags=False,
            node_size = 0.4,
            node_aspect = 2,
            node_label_size=12,
            link_label_fontsize=12,
            fig_ax=fig
            )
    else:
        figur, ax = fig
        figur.set_size_inches(4,4)
        if len(graph) > 0:
            tp.plot_graph(
            val_matrix=val_matrix,
            graph=graph,
            var_names=columnNames,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            show_autodependency_lags=False,
            node_size = 0.4,
            node_aspect = 1,
            node_label_size=12,
            link_label_fontsize=12,
            fig_ax=fig
            )
        else:
            matrixFull = np.zeros((val_matrix.shape[0], val_matrix.shape[1],2))
            matrixFull[:,:,1] = val_matrix
            tp.plot_graph(
            val_matrix=matrixFull,
            graph=matrixFull,
            var_names=columnNames,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            show_autodependency_lags=False,
            node_size = 0.4,
            node_aspect = 1,
            node_label_size=12,
            link_label_fontsize=12,
            fig_ax=fig
            )
    if save:
        plt.savefig(filename,dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def plotMatrix(val_matrix, columnNames, filename, show=False, save=True):
    if len(val_matrix.shape) == 3:
        val_matrix = absmaxND(val_matrix, axis=2)
    print(val_matrix)
    f1 = plt.figure(layout="constrained")
    f1.set_size_inches(6,5)
    g1 = sns.heatmap(val_matrix, annot=True, linewidth=0.5, xticklabels=columnNames, yticklabels=columnNames)
    g1.set(xlabel='To', ylabel='From')
    #plt.suptitle('GCSS', fontsize=23)
    if save:
        plt.savefig(dpi=200,fname=filename)

def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

def maxSignificantLink(x, bool_matr, axis = None):
    y = np.multiply(x, bool_matr)
    return absmaxND(y, axis = axis)

def getProjectionMap(lat, lon):
    if np.max(lat) > 85:
        return Basemap(projection ="npstere", boundinglat = np.min(lat),lon_0=0)
    if np.min(lat) < -85:
        return Basemap(projection = "spstere", boundinglat = np.max(lat),lon_0=0)
    else:
        R = 6371.0
        height = R * np.radians(np.max(lat) - np.min(lat))
        width = R * np.radians(np.max(lon) - np.min(lon)) * np.cos(np.radians(np.average(lat)))
        return Basemap(projection ="stere", width= width*1000, height= height*1000, lon_0 = np.average(lon), lat_0 = np.average(lat))

def showSpatialResults(args, spatialToSingle = [], singleToSpatial= [], save = True, show = False, saveNames =[], titles = [], spatialToSingleSignif = [], singleToSpatialSignif = []):
    sns.reset_defaults()

    lat = np.load(args.spatialLat)
    lon = np.load(args.spatialLon)

    fig = plt.figure(figsize=(4.5,5))
    map = getProjectionMap(lat, lon)
    x,y = map(lon, lat)
    minMax = np.max(np.abs(singleToSpatial))

    data = map.contourf(x,y,singleToSpatial,cmap= "bwr", levels = 128, vmin = -minMax, vmax = minMax, alpha=1)
    if len(singleToSpatialSignif) > 0:
        map.contourf(x,y,np.abs(singleToSpatialSignif) > 0, hatches=['', '////'], alpha=0)
    cb = map.colorbar(data, "bottom", size="5%", pad="2%", ticks=ticker.MaxNLocator(5))
    cb.ax.tick_params(labelsize=14)
    if len(titles) == 2:
        plt.title(titles[1],fontsize=15)
    map.drawcoastlines()
    if save and len(saveNames) == 2:
        plt.savefig(saveNames[1],dpi=300,bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig = plt.figure(figsize=(4.5,5))
    map = getProjectionMap(lat, lon)
    x,y = map(lon, lat)
    minMax = np.max(np.abs(spatialToSingle))
    data = map.contourf(x,y,spatialToSingle,cmap= "bwr", levels = 128, vmin = -minMax, vmax = minMax, alpha=1)
    if len(spatialToSingleSignif) > 0:
        map.contourf(x,y,np.abs(spatialToSingleSignif) > 0, hatches=['', '////'], alpha=0)
    cb = map.colorbar(data, "bottom", size="5%", pad="2%", ticks=ticker.MaxNLocator(5))
    cb.ax.tick_params(labelsize=14)
    if len(titles) == 2:
        plt.title(titles[0],fontsize=15)
    map.drawcoastlines()
    if save and len(saveNames) == 2:
        plt.savefig(saveNames[0],dpi=300,bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def showStationarityResults(filename, figTitle ="", oneToTwo = [], twoToOne= [], varNames = [], slideWindow = 20, startYear = 0, show = False, save = True):
    fig = plt.figure(figsize=(5,4),layout="constrained")
    xAxis = range(startYear + int(slideWindow/2), startYear + int(slideWindow/2) + len(oneToTwo), 1)
    plt.title(figTitle,fontsize=15)
    plt.xlabel("Central Year of " + str(slideWindow) + "-year Window",fontsize=14)
    plt.ylabel("Causal Effect Strength",fontsize=14)
    plt.locator_params(axis='x', nbins=5)
    plt.plot(xAxis, oneToTwo, label = varNames[0] + " To "+ varNames[1])
    plt.plot(xAxis, twoToOne, label = varNames[1] + " To "+ varNames[0])
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if save:
        plt.savefig(filename,dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

# working example: 
# py Pipeline.py -filename_in "./testSheet.csv" -dirname_out "testResults" -vars "SeaIce" "AMOC" "GrIS" -mask "mask_pcmci" "mask_lkif" "mask_pcmci" -mask_lkif "mask_lkif" -maskType "x" -detrend True True True -deseason True True True -diff True True True -tauMax 10 -alpha 0.1 -forbiddenLinks [0,2] -causalStationarity True -stationarityWindow 3 -stationarityShift 12 -stationarityVars 0 2

# py Pipeline.py -filename_in "./testSheet.csv" -dirname_out "testResults" -vars "SeaIce" "AMOC" "GrIS" 
# -mask "mask_pcmci" "mask_lkif" "mask_pcmci" -mask_lkif "mask_lkif" -maskType "x" 
# -detrend True True True -deseason True True True -diff True True True 
# -tauMax 10 -alpha 0.1 -forbiddenLinks [0,2] 
# -causalStationarity True -stationarityWindow 3 -stationarityShift 12 -stationarityVars 0 2

# working example for spatial analysis: 
# py Pipeline.py -filename_in "./testSheet.csv" -dirname_out "testResults" -vars "SeaIce" "AMOC" "GrIS" -mask "mask_pcmci" "mask_lkif" "mask_pcmci" -mask_lkif "mask_lkif" -maskType "x" -detrend True True True -deseason True True True -spatialAnalysis True -spatialFile testSeaIceAlbedo_Detrended.npy -spatialLon polarLongitudeAggregated.npy -spatialLat polarLatitudeAggregated.npy -spatialOtherVar 0

# py Pipeline.py -filename_in "./testSheet.csv" -dirname_out "testResults" -vars "SeaIce" "AMOC" "GrIS" 
# -mask "mask_pcmci" "mask_lkif" "mask_pcmci" -mask_lkif "mask_lkif" -maskType "x" 
# -detrend True True True -deseason True True True -spatialAnalysis True 
# -spatialFile testSeaIceAlbedo_Detrended.npy -spatialLon polarLongitudeAggregated.npy -spatialLat polarLatitudeAggregated.npy -spatialOtherVar 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-filename_in", type=str, help= ".csv file (separated by semicolons) with all univariate variables", required=True)
    parser.add_argument("-dirname_out", type=str, help= "directory for diagram output", required=True)
    parser.add_argument("-vars", help = "variable names in csv file", nargs='+', default=[], required=True)
    parser.add_argument("-mask", help = "mask column names in csv file (pcmci)", nargs='+', default=[], required=False)
    parser.add_argument("-mask_lkif", type = str, help = "single mask column name in csv file (lkif)", required=False)
    parser.add_argument("-maskType", type = str, choices = ["x","y", "z", "xyz"], default="x", required=False)
    parser.add_argument("-detrend", help = "detrend indicators (True/False), as many as vars", nargs='+', default=[], required=False)
    parser.add_argument("-deseason", help = "deseasonalizing indicators (True/False), as many as vars", nargs='+', default=[], required=False)
    parser.add_argument("-deseason_length", help ="deseason interval, i.e. 12 for months, 24 for hours etc.", default =12, required=False)
    parser.add_argument("-diff", help = "differentiation indicators (True/False), as many as vars", nargs='+', default=[], required=False)
    parser.add_argument("-tauMax", type=int, help = "maximal time lag for PCMCI, max. 24", default=5, required=False)
    parser.add_argument("-alpha", type=float, help = "error rate", default=0.05, required=False)
    parser.add_argument("-forbiddenLinks", help = "list of pairs with blocked connections from x1 to x2", nargs='+', default=[], required=False)
    parser.add_argument("-spatialAnalysis", help = "conduct spatial analysis, True/False", type=bool, default=False, required=False)
    parser.add_argument("-spatialFile", type=str, help = "npy file with spatial variable", required=False)
    parser.add_argument("-spatialLon", type=str, help = "longitude npy file", required=False)
    parser.add_argument("-spatialLat", type=str, help = "latitude npy file", required=False)
    parser.add_argument("-spatialOtherVar", type =int, help="index of other variable in spatial analysis", default=0, required=False)
    parser.add_argument("-causalStationarity", help = "conduct causal stationarity analysis, True/False", type=bool, default=False, required=False)
    parser.add_argument("-stationarityShift", type =int, help ="how far should the time window be shifted each time (for monthly data, 12 implies one year shift)", default=12, required=False)
    parser.add_argument("-stationarityWindow", type= int, help = "causal stationarity time window in number of shifts (e.g., number of years)", required=False)
    parser.add_argument("-stationarityVars", help = "indices of two variables for causal stationarity analysis", nargs = '+', default = [], required=False)
    parser.add_argument("-prescribeNetLM", help="user-prescribed network for linear mediation analysis", nargs='+', default = [], required = False)
    args = parser.parse_args()
    
    directory = args.dirname_out
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    except PermissionError:
        print(f"Permission denied: Unable to create directory '{directory}'.")
    except Exception as e:
        print(f"An error occurred trying to create '{directory}': {e}")
    #sys.stdout = open(directory + '/output.txt', 'w')

    # TODO: sanity checks on all arguments

    file_in = args.filename_in
    dataCols = args.vars
    maskCols = args.mask
    maskLKIF = args.mask_lkif
    if args.causalStationarity:
        stationaryShift = args.stationarityShift
        stationaryWindow = args.stationarityWindow
        a,b = args.stationarityVars
        a = ast.literal_eval(a)
        b = ast.literal_eval(b)
        slidingWindow = stationaryWindow * stationaryShift

    # default mask for lkif analysis (which can only take one mask)
    defaultMask = []
    useMask = True
    if len(maskCols) == 0:
        useMask = False
    else:
        maskDict = {}
        for i in range(len(maskCols)):
            maskName = maskCols[i]
            if maskName in maskDict.keys():
                maskDict[maskName].append(i)
            else:
                maskDict[maskName] = [i]
        maskKeys = list(maskDict.keys())
    try:
        dataCSV = pd.read_csv(file_in, sep=";")
        data = dataCSV[dataCols].to_numpy()
        mask = np.zeros(data.shape)
        defaultMask = np.zeros(data.shape[0])
        if useMask:
            maskEntries = dataCSV[maskKeys].to_numpy()
            if maskLKIF != "":
                defaultMask = dataCSV[maskLKIF].to_numpy()
            else:
                defaultMask = maskEntries[:,0]
            for i in range(len(maskCols)):
                mask[:,i] = maskEntries[:,maskKeys.index(maskCols[i])]
    except Exception as e:
        print(f"An error occurred trying to read '{file_in}': {e}")

    if len(args.forbiddenLinks) > 0:
        if len(args.prescribeNetLM) > 0:
            print("PCMCI or Linear Mediation recommended as they are the only methods handling forbidden links")
        else:
            print("PCMCI recommended as it is the only method handling forbidden links (and no network was given for Linear Mediation)")
    elif not useMask and data.shape[0] > 1000:
        print("Highest confidence in GCSS, combine methods")
    elif data.shape[0] < 200:
        print("Low sample count, low confidence in results, likely only identifies strong links correctly")
        if data.shape[1] > 10:
            print("LKIF recommended, potential performance problems with PCMCI")
        else:
            print("LKIF and PCMCI recommended")
    elif data.shape[1] > 10:
        print("Highest confidence in LKIF, potential performance problems with PCMCI")
    else: 
        print("Highest confidence in LKIF and PCMCI")
        

    inputStr = ""
    methodsChosen = []
    print("Methods available: LKIF, PCMCI, GCSS, LM")
    while inputStr != "end":
        inputStr = input("Enter desired method, enter \"end\" after last method: ")
        if inputStr == "end":
            continue
        methodsChosen.append(inputStr)
        print("Methods: ")
        print(methodsChosen)

    print(data)
    print(data.shape)
    print("Std Deviations (Non-normalized, non-masked):")
    for i in range(data.shape[1]):
        stdDev = np.std(data[:, i])
        print(dataCols[i] + ": " + str(stdDev))

    if useMask:
        if "PCMCI" in methodsChosen:
            print("Std Deviations (Non-normalized, masked for PCMCI):")
            for i in range(data.shape[1]):
                indices = mask[:,i] == 0
                stdDev = np.std(data[indices, i])
                print(dataCols[i] + ": " + str(stdDev))
        if "LKIF" in methodsChosen:
            print("Std Deviations (Non-normalized, masked for LKIF):")
            indices = defaultMask == 0
            for i in range(data.shape[1]):
                stdDev = np.std(data[indices, i])
                print(dataCols[i] + ": " + str(stdDev))

    if len(args.detrend) == len(dataCols):
        detrendingIndicator = args.detrend
    else:
        detrendingIndicator = np.zeros(len(dataCols))
    if len(args.deseason) == len(dataCols):
        deseasonIndicator = args.deseason
    else:
        deseasonIndicator = np.zeros(len(dataCols))
    if len(args.diff) == len(dataCols):
        diffIndicator = args.diff
    else:
        diffIndicator = np.zeros(len(dataCols))
    deseasonLength = args.deseason_length

    for i in range(len(dataCols)):
        existingData = data[:,i] != 0
        data[existingData,i] = (2 * ((data[existingData,i] - np.min(data[existingData,i])) / (np.max(data[existingData,i]) - np.min(data[existingData,i])))) - 1
        data[:,i] = np.reshape(preprocessData(np.reshape(data[:,i],(1,-1), order="F"), detrendingIndicator[i], deseasonIndicator[i], diffIndicator[i], deseasonLength=deseasonLength),(-1), order="F")
    
    tau_min = 1
    tau_max = args.tauMax
    alpha = args.alpha

    if "PCMCI" in methodsChosen:
        independenceTest = input("Select a correlation test for PCMCI\n from \"parcorr\", \"robustparcorr\", \"GPDC\", \"parcorrWLS\": ")
        if independenceTest == "parcorr":
            ci_test = ParCorr(significance='analytic', mask_type=None if (not useMask) else args.maskType)
        elif independenceTest == "robustparcorr":
            ci_test = RobustParCorr(significance='analytic', mask_type=None if (not useMask) else args.maskType)
        elif independenceTest == "GPDC":
            ci_test = GPDC(significance = 'analytic', mask_type=None if (not useMask) else args.maskType)
        elif independenceTest == "parcorrWLS":
            ci_test = ParCorrWLS(significance = 'analytic', mask_type=None if (not useMask) else args.maskType)
        else:
            print("No independence Test selected, defaulting to parCorr")
            ci_test = ParCorr(significance='analytic', mask_type=None if (not useMask) else args.maskType)

    linkAssumptions = {}
    for i in range(len(dataCols)):
        linkAssumptions[i] = {}
        for j in range(len(dataCols)):
            for lag in range(tau_min, tau_max+1):
                linkAssumptions[i][(j,-lag)] = "o?o"

    if len(args.forbiddenLinks) > 0:
        forbLi = np.array([ast.literal_eval(item) for item in args.forbiddenLinks])
        print(forbLi)
        for (u,v) in forbLi:
            for lag in range(tau_min, tau_max+1):
                linkAssumptions[v].pop((u,-lag))

    if not args.spatialAnalysis:
        
        if "PCMCI" in methodsChosen:
            if useMask:
                dataframe = pp.DataFrame(data, var_names=dataCols, mask = mask)
            else:
                dataframe = pp.DataFrame(data, var_names = dataCols)
            pcmci = PCMCI(
                dataframe=dataframe, 
                cond_ind_test=ci_test,
                verbosity=1)
                
            results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=alpha, reset_lagged_links=True, link_assumptions = linkAssumptions)

            val_matrix = results["val_matrix"]
            p_matrix = results["p_matrix"]
            graph_bool = p_matrix <= alpha
            matrixPCMCI = val_matrix * graph_bool
            
            visualizeGraph(matrixPCMCI, results["graph"], dataCols, directory + "/default_pcmci")

            if useMask:
                med = LinearMediation(dataframe=dataframe, mask_type = args.maskType)
            else:
                med = LinearMediation(dataframe=dataframe, mask_type = None)
            med.fit_model(all_parents=toys.dag_to_links(results["graph"]), tau_max=tau_max)
            val_matrix = med.get_val_matrix(symmetrize=True)

            visualizeGraph(val_matrix, results["graph"], dataCols, directory + "/default_mediatedPCMCI")

            print("Linear Mediation Values:")
            pcmci.print_significant_links(
                        p_matrix = results["p_matrix"],
                        val_matrix = val_matrix,
                        alpha_level = alpha)
            
            if args.causalStationarity:
                oneToTwo = np.zeros(int((data.shape[0]-slidingWindow)/stationaryShift))
                twoToOne = np.zeros(int((data.shape[0]-slidingWindow)/stationaryShift))
                for i in range(int((data.shape[0]-slidingWindow)/stationaryShift)):
                    dataframe = pp.DataFrame(data[i*stationaryShift:i*stationaryShift+slidingWindow], var_names=dataCols, mask = mask[i*stationaryShift:i*stationaryShift+slidingWindow])
                    pcmci = PCMCI(
                        dataframe=dataframe, 
                        cond_ind_test=ci_test,
                        verbosity=0)
                    results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=alpha, reset_lagged_links=True, link_assumptions = linkAssumptions)
                    val_matrix = results["val_matrix"]
                    p_matrix = results["p_matrix"]
                    graph_bool = p_matrix <= alpha
                    matrixPCMCI = maxSignificantLink(val_matrix, graph_bool, axis = 2)
                    oneToTwo[i] = matrixPCMCI[a,b]
                    twoToOne[i] = matrixPCMCI[b,a]
                showStationarityResults(directory + "/stationarity_pcmci", "", oneToTwo, twoToOne, varNames=[dataCols[a], dataCols[b]], slideWindow= stationaryWindow)
        
        if "LM" in methodsChosen:
            if len(args.prescribeNetLM) == 0:
                print("Error in Linear Mediation: No user defined network given")
                exit()
            else:
                lmLinkAssumptions = {}
                for u in range(data.shape[1]):
                    lmLinkAssumptions.setdefault(u, [])
                givenLinks = np.array([ast.literal_eval(item) for item in args.prescribeNetLM])
                for (u,v,time) in givenLinks:
                    lmLinkAssumptions[v].append((u,-time))
                if useMask:
                    dataframe = pp.DataFrame(data, var_names=dataCols, mask = mask)
                else:
                    dataframe = pp.DataFrame(data, var_names = dataCols)
                if useMask:
                    med = LinearMediation(dataframe=dataframe, mask_type = args.maskType)
                else:
                    med = LinearMediation(dataframe=dataframe, mask_type = None)
                med.fit_model(all_parents=lmLinkAssumptions, tau_max=tau_max)
                val_matrix = med.get_val_matrix(symmetrize=True)

                visualizeGraph(val_matrix, results["graph"], dataCols, directory + "/default_linearMed")

                if args.causalStationarity:
                    oneToTwo = np.zeros(int((data.shape[0]-slidingWindow)/stationaryShift))
                    twoToOne = np.zeros(int((data.shape[0]-slidingWindow)/stationaryShift))
                    for i in range(int((data.shape[0]-slidingWindow)/stationaryShift)):
                        dataframe = pp.DataFrame(data[i*stationaryShift:i*stationaryShift+slidingWindow], var_names=dataCols, mask = mask[i*stationaryShift:i*stationaryShift+slidingWindow])
                        if useMask:
                            med = LinearMediation(dataframe=dataframe, mask_type = args.maskType)
                        else:
                            med = LinearMediation(dataframe=dataframe, mask_type = None)
                        med.fit_model(all_parents=lmLinkAssumptions, tau_max=tau_max)
                        val_matrix = med.get_val_matrix(symmetrize=True)
                        val_matrix = absmaxND(val_matrix, axis=2)
                        oneToTwo[i] = val_matrix[a,b]
                        twoToOne[i] = val_matrix[b,a]
                    showStationarityResults(directory + "/stationarity_linearMed", "", oneToTwo, twoToOne, varNames=[dataCols[a], dataCols[b]], slideWindow= stationaryWindow)

        if "LKIF" in methodsChosen:
            if useMask:
                maskSelector = defaultMask == 0
                indexArr = np.array([i for i in range(mask.shape[0])])
                matrixLKIF = LKIF.lkif(data[maskSelector].T, alpha, returnAll=False, timestamps = indexArr[maskSelector])
            else:
                matrixLKIF = LKIF.lkif(data.T, alpha, returnAll=False, timestamps = None)
            
            visualizeGraph(matrixLKIF, [], dataCols, directory + "/default_lkif")

            if args.causalStationarity:
                oneToTwo = np.zeros(int((data.shape[0]-slidingWindow)/stationaryShift))
                twoToOne = np.zeros(int((data.shape[0]-slidingWindow)/stationaryShift))
                for i in range(int((data.shape[0]-slidingWindow)/stationaryShift)):
                    localData = data[i*stationaryShift:i*stationaryShift+slidingWindow]
                    if useMask:
                        maskSelector = defaultMask[i*stationaryShift:i*stationaryShift+slidingWindow] == 0
                        indexArr = np.array([i for i in range(localData.shape[0])])
                        matrixLKIF = LKIF.lkif(localData[maskSelector].T, alpha, returnAll=False, timestamps = indexArr[maskSelector])
                    else:
                        matrixLKIF = LKIF.lkif(localData.T, alpha, returnAll=False, timestamps = None)
                        
                    oneToTwo[i] = matrixLKIF[a,b]
                    twoToOne[i] = matrixLKIF[b,a]
                showStationarityResults(directory + "/stationarity_lkif", "", oneToTwo, twoToOne, varNames=[dataCols[a], dataCols[b]], slideWindow= stationaryWindow)

        if "GCSS" in methodsChosen:
            try:
                matrixGCSS = GCSS.gcss(data.T, alpha, tau_max, returnAll=False).T
                visualizeGraph(matrixGCSS, [], dataCols, directory + "/default_gcss")
            except:
                matrixGCSS = np.zeros((data.shape[1],data.shape[1]))
                print("Error on GCSS")
            
            if args.causalStationarity:
                oneToTwo = np.zeros(int((data.shape[0]-slidingWindow)/stationaryShift))
                twoToOne = np.zeros(int((data.shape[0]-slidingWindow)/stationaryShift))
                for i in range(int((data.shape[0]-slidingWindow)/stationaryShift)):
                    localData = data[i*stationaryShift:i*stationaryShift+slidingWindow]
                    try:
                        matrixGCSS = GCSS.gcss(localData.T, alpha, tau_max, returnAll=False).T
                    except:
                        matrixGCSS = np.zeros((localData.shape[1],localData.shape[1]))
                        print("Error on GCSS")
                        
                    oneToTwo[i] = matrixGCSS[a,b]
                    twoToOne[i] = matrixGCSS[b,a]
                showStationarityResults(directory + "/stationarity_gcss", "", oneToTwo, twoToOne, varNames=[dataCols[a], dataCols[b]], slideWindow= stationaryWindow)
# spatial Analysis
    else:
        data_spatialResolved = np.load(args.spatialFile)
        data_final = np.zeros((data.shape[0], data.shape[1]+1))
        data_final[:,0:data.shape[1]] = data
        dataCols_extended = dataCols + ["Spatial Data"]
        singleVar = args.spatialOtherVar
        if "PCMCI" in methodsChosen:
            spatialToSinglePCM = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
            singleToSpatialPCM = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
        if "GCSS" in methodsChosen:
            spatialToSingleGCSS = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
            singleToSpatialGCSS = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
            spatialToSingleSignifGCSS = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
            singleToSpatialSignifGCSS = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
        if "LKIF" in methodsChosen:
            spatialToSingleLKIF = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
            singleToSpatialLKIF = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
            spatialToSingleSignifLKIF = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
            singleToSpatialSignifLKIF = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
        if "LM" in methodsChosen:
            spatialToSingleLM = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
            singleToSpatialLM = np.zeros((data_spatialResolved.shape[1],data_spatialResolved.shape[2]))
            if len(args.prescribeNetLM) == 0:
                print("Error in Linear Mediation: No user defined network given")
                exit()
            else:
                lmLinkAssumptions = {}
                lmLag = -1
                givenLinks = np.array([ast.literal_eval(item) for item in args.prescribeNetLM])
                for (u,v,time) in givenLinks:
                    lmLinkAssumptions.setdefault(v, []).append((u,-time))
                    if (u == singleVar and v == data.shape[1]) or (v == singleVar and u == data.shape[1]):
                        lmLag = time
                if lmLag == -1:
                    print("Error in Linear Mediation: Prescribed network doesn't contain spatially analyzed link")

        existingEntries = []
        #for LKIF
        maskSelector = defaultMask == 0
        indexArr = np.array([i for i in range(len(mask))])
        for i in range(data_spatialResolved.shape[1]):
            for j in range(data_spatialResolved.shape[2]):
                col = data_spatialResolved[:data.shape[0],i,j]
                if np.isnan(col[0]): 
                    continue
                if np.all(col == 0):
                    continue
                data_final[:,data_final.shape[1]-1] = col[:]
                data_final[:,data_final.shape[1]-1] = (2 * ((data_final[:,data_final.shape[1]-1] - np.min(data_final[:,data_final.shape[1]-1])) / (np.max(data_final[:,data_final.shape[1]-1]) - np.min(data_final[:,data_final.shape[1]-1])))) - 1
                if "GCSS" in methodsChosen:
                    existingEntries.append([data, data_spatialResolved[:data.shape[0], i,j]])

                if "PCMCI" in methodsChosen:
                    dataframe = pp.DataFrame(data_final, var_names=dataCols_extended, mask = mask)
                    pcmci = PCMCI(
                    dataframe=dataframe, 
                    cond_ind_test=ci_test,
                    verbosity=0)
                    results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=alpha, reset_lagged_links=True, link_assumptions = linkAssumptions)
                    val_matrix = results["val_matrix"]
                    p_matrix = results["p_matrix"]
                    graph_bool = p_matrix <= alpha
                    matrixPCMCI = maxSignificantLink(val_matrix, graph_bool, axis = 2)
                    spatialToSinglePCM[i,j] = matrixPCMCI[data.shape[1],singleVar]
                    singleToSpatialPCM[i,j] = matrixPCMCI[singleVar,data.shape[1]]

                if "LM" in methodsChosen:
                        if useMask:
                            dataframe = pp.DataFrame(data_final, var_names=dataCols, mask = mask)
                        else:
                            dataframe = pp.DataFrame(data_final, var_names = dataCols)
                        if useMask:
                            med = LinearMediation(dataframe=dataframe, mask_type = args.maskType)
                        else:
                            med = LinearMediation(dataframe=dataframe, mask_type = None)
                        med.fit_model(all_parents=lmLinkAssumptions, tau_max=tau_max)
                        val_matrix = med.get_val_matrix(symmetrize=True)
                        spatialToSingleLM[i,j] = med.get_coeff(i=data.shape[1], tau=-lmLag, j=singleVar)
                        singleToSpatialLM[i,j] = med.get_coeff(i=singleVar, tau=-lmLag, j=data.shape[1])
                if "LKIF" in methodsChosen:
                    if useMask:
                        try:
                            matrixLKIF, signif = LKIF.lkif(data_final[maskSelector].T, alpha, returnAll=True, timestamps = indexArr[maskSelector], returnSignif=True)
                            #matrixLKIF = LKIF.lkif(data_final[maskSelector].T, alpha, returnAll=False, timestamps = indexArr[maskSelector])
                        except Exception as e:
                            print(e)
                            matrixLKIF = np.zeros((data_final.shape[1],data_final.shape[1]))
                            signif = np.zeros((data_final.shape[1],data_final.shape[1]))
                    else:
                        try:
                            matrixLKIF, signif = LKIF.lkif(data_final.T, alpha, returnAll=True, timestamps = None, returnSignif=True)
                        except:
                            print("Error on LKIF")
                            matrixLKIF = np.zeros((data_final.shape[1],data_final.shape[1]))
                            signif = np.zeros((data_final.shape[1],data_final.shape[1]))
                    matrixLKIF[np.isnan(matrixLKIF)] = 0
                    matrixLKIF = np.maximum(matrixLKIF, 0)
                    spatialToSingleLKIF[i,j] = matrixLKIF[data.shape[1],singleVar]
                    singleToSpatialLKIF[i,j] = matrixLKIF[singleVar,data.shape[1]]
                    spatialToSingleSignifLKIF[i,j] = signif[data.shape[1],singleVar]
                    singleToSpatialSignifLKIF[i,j] = signif[singleVar,data.shape[1]]

        if "PCMCI" in methodsChosen:
            showSpatialResults(args, spatialToSinglePCM, singleToSpatialPCM, saveNames = [directory + "/spatialToSingle_pcmci", directory + "/singleToSpatial_pcmci"])    
        if "LKIF" in methodsChosen:                
            showSpatialResults(args, spatialToSingleLKIF, singleToSpatialLKIF, saveNames = [directory + "/spatialToSingle_lkif", directory + "/singleToSpatial_lkif"],
                                       spatialToSingleSignif=spatialToSingleSignifLKIF, singleToSpatialSignif=singleToSpatialSignifLKIF)
        if "LM" in methodsChosen:
            showSpatialResults(args, spatialToSingleLM, singleToSpatialLM, saveNames = [directory + "/spatialToSingle_lm", directory + "/singleToSpatial_lm"])
            if "PCMCI" in methodsChosen:
                showSpatialResults(args, spatialToSingleLM, singleToSpatialLM, saveNames = [directory + "/spatialToSingle_pcmciWithLM", directory + "/singleToSpatial_pcmciWithLM"],
                                   spatialToSingleSignif=spatialToSinglePCM, singleToSpatialSignif=singleToSpatialPCM)
        
        if "GCSS" in methodsChosen:
            x = np.array(existingEntries)
            matrices, signif = GCSS.gcssBulk(x,alpha=alpha, tau_max =tau_max)
            counter = 0
            for i in range(data_spatialResolved.shape[1]):
                for j in range(data_spatialResolved.shape[2]):
                    col = data_spatialResolved[:data_final.shape[0],i,j]
                    if np.isnan(col[0]): 
                        continue
                    if np.all(col == 0):
                        continue
                    if matrices.shape[0] > counter:
                        spatialToSingleGCSS[i,j] = matrices[counter, singleVar,data.shape[1]]
                        singleToSpatialGCSS[i,j] = matrices[counter, data.shape[1],singleVar]
                        spatialToSingleSignifGCSS[i,j] = signif[counter, singleVar,data.shape[1]]
                        singleToSpatialSignifGCSS[i,j] = signif[counter, data.shape[1],singleVar]
                    else: 
                        print("Error: ran out of values in GCSS return values")
                        break
                    counter = counter+1
            showSpatialResults(args, spatialToSingleGCSS, singleToSpatialGCSS, saveNames = [directory + "/spatialToSingle_gcss", directory + "/singleToSpatial_gcss"],
                                       spatialToSingleSignif=spatialToSingleSignifGCSS, singleToSpatialSignif=singleToSpatialSignifGCSS)