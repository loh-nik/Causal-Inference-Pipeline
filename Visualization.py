from matplotlib import pyplot as plt
import numpy as np
import dtreeviz
from dtreeviz import decision_boundaries
from sklearn.tree import DecisionTreeClassifier
import networkx as nx
import seaborn as sns
from tigramite import plotting as tp

def saveROCCurve(TPR, FPR, values, title, filename, colors = [], rowLabels = [], show = False, save=True, annotateBest = True,
                 xscale = "linear", yscale = "linear", figsize=(4.5,4), dpi=100, xlabel ="", ylabel =""):
    fig = plt.figure(figsize=figsize, layout="constrained")
    #plt.suptitle(title, fontsize=15)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if TPR.shape != FPR.shape:
        print("Error: TPR and FPR shaped differently")
        exit()
    if TPR.shape[0] != len(values) and TPR.shape[1] != len(values):
        print("Value list doesn't match any dimension of data")
        exit()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05,1.05)
    # should be int for one-dimensional data, should be array of ints for two-dim data
    showIndices = np.argmin(np.power(1-TPR, 2) + np.power(FPR,2), axis=1 if len(TPR.shape) > 1 else 0)
    if len(TPR.shape) == 1:
        plt.plot(FPR, TPR)
        if annotateBest:
            plt.annotate(values[showIndices], (FPR[showIndices], TPR[showIndices]),fontsize=14)
    else:
        showLegend = True
        if len(rowLabels) == 0:
            rowLabels = np.zeros(TPR.shape[0])
            showLegend = False
        elif len(rowLabels) != TPR.shape[0]:
            print("Error: Not enough labels provided for TPR/FPR data")
            exit()
        if len(colors) > 0 and len(colors) != TPR.shape[0]:
            print("Error: Not enough colors provided for TPR/FPR data")
            exit()
        for i in range(TPR.shape[0]):
            if len(colors) == 0:
                plt.plot(FPR[i], TPR[i], label = str(rowLabels[i]))
                if annotateBest:
                    plt.plot(FPR[i, showIndices[i]], TPR[i, showIndices[i]], 'o', color="black")
            else:
                plt.plot(FPR[i], TPR[i], label = str(rowLabels[i]), color=colors[i])
                if annotateBest:
                    plt.plot(FPR[i, showIndices[i]], TPR[i, showIndices[i]], 'o', color=colors[i])
            if annotateBest:
                plt.annotate(values[showIndices[i]], (FPR[i, showIndices[i]], TPR[i, showIndices[i]]),fontsize=14)
        if showLegend:
            plt.legend(fontsize=14)
    if save:
        plt.savefig(filename,dpi = dpi)
    if show:
        plt.show()
    else:
        plt.close()

def saveF1Curve(scores, values, title, filename, errors = [], colors = [], rowLabels=[], show = False, save= True, 
                xscale = "linear", yscale = "linear", figsize=(4.5,4), dpi=200, xlabel ="", ylabel ="", yAxisCut = False, yAxisLinearLim = 1):
    fig = plt.figure(figsize=figsize, layout ='constrained')
    if scores.shape[0] != len(values) and scores.shape[1] != len(values):
        print("Value list doesn't match any dimension of data")
        exit()
    if len(errors) > 0 and errors.shape != scores.shape:
        print("Errors shape doesn't match scores shape")
        exit()
    #plt.suptitle(title, fontsize=15)
    plt.xscale(xscale)
    if yAxisCut:
        plt.yscale("symlog", linthresh = yAxisLinearLim)
        plt.yticks(np.append(np.arange(0,1,0.1), range(1,8)), labels=["0","","","","","0.5","","","","","1","","","","5","",""])
        ax = plt.gca()
        ax.grid(True, axis="y", color='lightgray', linestyle='-', linewidth=0.5)
        ax.axhline(y=yAxisLinearLim, color='gray', linewidth=1, linestyle='-')
    else:
        plt.yscale(yscale)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if len(scores.shape) == 1:
        plt.plot(values, scores)
        if len(errors) > 0:
            plt.fill_between(values, scores + errors, scores - errors, alpha = 0.2)
    else:
        showLegend = True
        if len(rowLabels) == 0:
            rowLabels = np.zeros(scores.shape[0])
            showLegend = False
        elif len(rowLabels) != scores.shape[0]:
            print("Error: Not enough labels provided for Score data")
            exit()
        if len(colors) > 0 and len(colors) != scores.shape[0]:
            print("Error: Not enough colors provided for Score data")
            exit()
        for i in range(scores.shape[0]):
            if len(colors) == 0:
                plt.plot(values, scores[i], label = str(rowLabels[i]))
                if len(errors) > 0:
                    plt.fill_between(values, scores[i] + errors[i], scores[i] - errors[i], alpha = 0.2)
            else:
                plt.plot(values, scores[i], label = str(rowLabels[i]), color=colors[i])
                if len(errors) > 0:
                    plt.fill_between(values, scores[i] + errors[i], scores[i] - errors[i], alpha = 0.2, color= colors[i])
        if showLegend:
            plt.legend(fontsize=14)
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

def saveDecisionTree(data, labels, feature_names, class_names, filename, max_depth = 3, show = False, save = True):
    clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
    clf = clf.fit(data, labels)
    viz_model = dtreeviz.model(clf,
                                   X_train = data, y_train=labels,
                                   feature_names = feature_names,
                                   target_name = "Ranking", class_names = class_names)
    v = viz_model.view(fancy=False)
    if save:
        v.save(filename)
    if show:
        v.show()

def saveDecisionBoundaries(data, labels, filename, show = False, save = True):
    if data.shape[1] != 2 :
        print("Invalid data to draw 2d decision boundaries")
        exit()
    if np.max(labels) == 0:
        print("Only one class found, no decision boundary can be drawn")
        return
    fix, ax = plt.subplots()
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf = clf.fit(data, labels)
    decision_boundaries(clf, data, labels, ax = ax)
    if save:
        plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()

def saveCouplingMatrixGraph(matrix, title, filename, show = False, save= True, figsize=(4,4.5), dpi=100):
    ax = plt.subplots(1,1,layout="constrained", figsize=figsize)
    plt.title(title, fontsize=17)
    #G = nx.from_numpy_array(matrix, create_using =nx.DiGraph)
    matrixFull = np.zeros((matrix.shape[0], matrix.shape[1],2))
    matrixFull[:,:,1] = matrix
    tp.plot_graph(val_matrix=matrixFull,
            graph=matrixFull,
            show_colorbar=False,
            var_names=range(matrixFull.shape[0]),
            show_autodependency_lags=False,
            fig_ax = ax,
            node_aspect = 1,
            node_label_size=14,
            link_label_fontsize = 1
            )
    #G = nx.from_numpy_array(matrix, create_using =nx.DiGraph)
    #nx.draw(G, with_labels=True, font_weight='bold')
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else: 
        plt.close()

def saveHeatmap(textValues, colorValues, title, filename, show=False, save=True, figsize = (4.5,4), dpi=200, xlabel= "", ylabel ="", xtickLabels = [], ytickLabels=[]):
    fig = plt.figure(figsize=figsize, layout="constrained")
    #plt.suptitle(title, fontsize=15)
    ax = sns.heatmap(colorValues, cmap = "Wistia", annot = textValues, annot_kws={"fontsize": 14})
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=12)
    ax.invert_yaxis()
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if len(xtickLabels) == 0:
        plt.xticks(fontsize=14)
    else:
        plt.xticks(ticks=np.array(range(len(xtickLabels))) + 0.5, labels=xtickLabels, fontsize=12)
    if len(ytickLabels) == 0:
        plt.yticks(fontsize=14)
    else:
        plt.yticks(ticks=np.array(range(len(ytickLabels))) + 0.5, labels=ytickLabels,fontsize=12)
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else: 
        plt.close()

def saveGrid(matrix, title, filename, show=False, save = True, figsize=(4.2,3.8), dpi=300):
    fig, ax = plt.subplots(1,1,layout="constrained", figsize=figsize)
    #plt.suptitle(title,fontsize=15)
    maxVal = np.max(np.abs(matrix))
    cax = ax.imshow(matrix, cmap='bwr', vmin = -maxVal, vmax = maxVal)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(cax,shrink=0.9)
    cb.ax.tick_params(labelsize=14)
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else: 
        plt.close()