import tigramite 
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gpdc import GPDC

def PCMCIPlus(numeric_data, mask, var_names, mask_type, tauMax = 5, alpha = 0.05, contempLinks = True, independenceTest = "parcorr"):
    if mask != []:
        dataframe = pp.DataFrame(numeric_data, var_names=var_names, mask = mask)
    else:
        dataframe =  pp.DataFrame(numeric_data, var_names=var_names)

    if independenceTest == "parcorr":
        parcorr = ParCorr(significance='analytic', mask_type=mask_type)
    elif independenceTest == "robustparcorr":
        parcorr = RobustParCorr(significance='analytic', mask_type=mask_type)
    elif independenceTest == "GPDC":
        parcorr = GPDC(significance = 'analytic', mask_type=mask_type)
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=parcorr,
        verbosity=0)

    tau_min = 0 if contempLinks else 1
    tau_max = tauMax
    pc_alpha = alpha

    # default pcmci plus
    results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha, reset_lagged_links=True)

    return results['val_matrix'], results['p_matrix']