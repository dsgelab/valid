# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from utils import *
# Standard stuff
import numpy as np
import pandas as pd
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
from datetime import datetime
# Plotting
import seaborn as sns
import matplotlib
matplotlib.use("Agg")

sns.set_style("whitegrid")
# Statistics/Processing
import scipy.stats as st
from processing_utils import *

def remove_severe_value_outliers(data, max_z, n_indvs_stats, res_dir, file_name, plot=True):
    """Removes severe outliers based on the z-scores >= 10. Saving plots before and after."""

    out_dir = res_dir + "plots/"; make_dir(out_dir)
    ### Severe z-score outliers
    data = data.assign(Z=st.zscore(data.VALUE))

    # Make sure not to include single extreme outliers with the value quant. I.e. LDL had one such individual completely skewing the upper box.
    data = data.assign(VALUE_QUANT=pd.qcut(data.loc[np.abs(data.Z) < 100].VALUE + (data.loc[np.abs(data.Z) < 100].VALUE+0.01*(np.random.rand(data.loc[np.abs(data.Z)<100].shape[0])-0.5)), 100000).apply(lambda x: x.mid))
    data = data.assign(VALUE_QUANT=data.VALUE_QUANT.astype("float").round())
    stat_value_quants = data.groupby("VALUE_QUANT").agg({"FINNGENID": lambda x: len(set(x))}).reset_index().rename(columns={"FINNGENID": "N_INDVS"}).copy()
    logger.info("Min value quant counts: {}".format(stat_value_quants.N_INDVS.min()))
    # Log with plots
    if plot:
        plt.figure()
        fig, ax = plt.subplots(1,2, figsize=(5,5))
        sns.boxplot(data, y="VALUE_QUANT",ax=ax[0])
        sns.boxplot(data.loc[np.abs(data.Z)<max_z], y="VALUE_QUANT", ax=ax[1])
        ax[0].set_ylabel("Value")
        ax[1].set_ylabel("")
        plt.savefig(out_dir+file_name+"_valuequants.pdf")
        plt.savefig(out_dir+file_name+"_valuequants.png")

        stat_value_quants.to_csv(out_dir+file_name+"_valuequants.csv", sep=",", index=False)
        print("start")
        plt.figure()
        fig, ax = plt.subplots(1,2,figsize=(5,5))
        sns.scatterplot(data, x="EVENT_AGE", y="VALUE", ax=ax[0], hue="Z")
        sns.scatterplot(data.loc[np.abs(data.Z)<max_z], x="EVENT_AGE", y="VALUE", ax=ax[1], hue="Z")
        plt.savefig(out_dir+file_name+"_zs.png")
        print("end")
    # Stats
    n_indv = len(set(data.FINNGENID))
    n_row_remove = data.loc[np.abs(data.Z) >= max_z].shape[0]
    logger.info("Max z of {} removed values greater than {} and lower than {}".format(max_z, data.loc[data.Z >= max_z].VALUE.min(), data.loc[data.Z <= -max_z].VALUE.max()))
    # Removing
    data = data.loc[np.abs(data.Z) < max_z]
    # Stats
    n_indv_remove = n_indv - len(set(data.FINNGENID))
    n_indv = len(set(data.FINNGENID))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers","N_INDVS_NOW"] = n_indv 
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers","N_INDVS_REMOVED"] = n_indv_remove
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers","N_ROWS_REMOVED"] = n_row_remove

    return(data, n_indvs_stats)

def remove_single_value_outliers(data, n_indvs_stats):
    # Z-scores for each individuals value distributions
    data = data.assign(Z_indv=data.groupby("FINNGENID").VALUE.transform(lambda x: st.zscore(x)))
    # Outliers
    outliers_high = data.loc[data.Z_indv > 2.5].groupby("FINNGENID").filter(lambda x: len(x) == 1).query("Z_indv>5 & VALUE>500")
    outliers_low = data.loc[data.Z_indv < -4].groupby("FINNGENID").filter(lambda x: len(x) == 1).query("VALUE<53")
    outliers = pd.concat([outliers_high, outliers_low])
    # Remove
    data = data.drop(outliers.index).copy()
    # Stats   
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_INDVS_NOW"] =  len(set(data.FINNGENID)) 
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_ROWS_REMOVED"] = outliers.shape[0]
    data = data.drop(columns=["Z_indv"])

    return(data, n_indvs_stats)
    
def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Path to extracted file (step0)", required=True)
    parser.add_argument("--res_dir", help="Path to the results directory (step1)", default="/home/ivm/valid/data/processed_data/step1/")
    parser.add_argument("--lab_name", help="Readable name of the measurement value.", required=True)
    parser.add_argument("--max_z", type=int, help="Maximum z-score among all measurements. [dafult: 10]", default=10)
    parser.add_argument("--plot", type=int, help="Minimum reasonable value [dafult: None]", default=1)

    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    #### Preparations
    timer = Timer()
    args = get_parser_arguments()
    
    file_name = args.lab_name + "_" + get_date() 
    count_dir = args.res_dir + "counts/"
    log_file_name = args.lab_name +  "_" + get_datetime()
    make_dir(args.res_dir); make_dir(count_dir)
    
    init_logging(args.res_dir, log_file_name, logger, args)

    #### Data processing
    # Raw data
    data = pd.read_csv(args.file_path, sep="\t", compression="gzip")
    # Prep
    n_indvs_stats = pd.DataFrame({"STEP": ["Start", "Outliers", "Outliers_single"]})
    n_indv = len(set(data.FINNGENID))

    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_INDVS_NOW"] = n_indv

    data = data[["FINNGENID", "SEX", "EVENT_AGE", "APPROX_EVENT_DATETIME", "egfr_ckdepi2021"]].copy()
    data = data.rename(columns={"APPROX_EVENT_DATETIME": "DATE", "egfr_ckdepi2021": "VALUE"})
    data, n_indvs_stats = remove_severe_value_outliers(data, args.max_z, n_indvs_stats, args.res_dir, file_name, args.plot)
    print(n_indvs_stats)

    data, n_indvs_stats = remove_single_value_outliers(data, n_indvs_stats)
    print(n_indvs_stats)

    #### Finishing
    data = get_abnorm_func_based_on_name(args.lab_name)(data, "VALUE")
    logging.info("Min abnorm " + str(data.loc[data.ABNORM_CUSTOM == 1.0].VALUE.min()) + " max abnorm " + str(data.loc[data.ABNORM_CUSTOM == 1.0].VALUE.max()))
    
    # Saving
    data = data.drop(columns=["Z", "VALUE_QUANT"])
    data.to_csv(args.res_dir + file_name + ".csv", sep=",", index=False)
    n_indvs_stats.to_csv(count_dir + file_name + "_counts.csv", sep=",", index=False)
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
