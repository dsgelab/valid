# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/"))
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
sns.set_style("whitegrid")
# Statistics/Processing
import scipy.stats as st


"""Get only entries with the main unit. Records new and removed number of rows and individuals in statistics df n_indvs_stats."""
def get_main_unit_data(data, n_indvs_stats, main_unit):
    # Stats
    n_indv = len(set(data.FINNGENID))
    n_row_remove = data.loc[data.UNIT != main_unit].shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_ROWS_REMOVED"] = n_row_remove
    # Removing
    data = data.loc[data.UNIT == main_unit]
    # Stats
    n_indv_remove = abs(len(set(data.FINNGENID))-n_indv)
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_INDVS_REMOVED"] = n_indv_remove
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_INDVS_NOW"] = len(set(data.FINNGENID))
       
    return(data, n_indvs_stats)
    
"""Removes rows without any measurement value. Records number of rows and individuals in statistics df n_indvs_stats."""
def remove_missing_values(data, n_indvs_stats):
    # Stats
    n_indv = len(set(data.FINNGENID))
    n_row_remove = data["VALUE"].isnull().sum()
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_ROWS_REMOVED"] = n_row_remove
    # Removing
    data = data.loc[data["VALUE"].notnull()]
    # Stats
    n_indv_remove = abs(len(set(data.FINNGENID))-n_indv)
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_INDVS_REMOVED"] = n_indv_remove
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_INDVS_NOW"] = len(set(data.FINNGENID))
       
    return(data, n_indvs_stats)

"""Duplicates are measurements at exact same time. Handling so that same value measurements, only
    one is kept. For measurements with different values, removing all measurements at the same
    time where the difference between min and max > 10%IQR of all measurements. As it is not
    possible to know what is causing this extreme difference. 
    Averaging the rest and recording the max-flux value for this test.
    Records number of rows and individuals in statistics df n_indvs_stats."""
def handle_duplicates(data, n_indvs_stats):
    # Stats
    n_indv = len(set(data.FINNGENID))
    n_rows = data.shape[0]
    ### Duplicates based on exact same time
    dups = data[data.duplicated(subset=["FINNGENID", "DATE"], keep=False)].reset_index(drop=True)
    data = data.drop_duplicates(["FINNGENID", "DATE"], keep=False).reset_index(drop=True)
    logger.info("{:,} measurements at the exact same time".format(len(dups)))
        
    ### Max flux removing measurements
    max_flux = (data["VALUE"].describe()["75%"]-data["VALUE"].describe()["25%"])*0.1
    # Stats
    n_row_remove = len(dups.groupby(["FINNGENID", "DATE"]).filter(lambda x: np.ptp(x["VALUE"]) > max_flux))
    logger.info("Removing {:,} measurements at the exact same time with distance greater max flux ({})".format(n_row_remove, max_flux))
    # Removing
    dups = dups.groupby(["FINNGENID", "DATE"]).filter(lambda x: np.ptp(x["VALUE"]) <= max_flux) 
    
    ### Averaging rest
    dups = dups.groupby(["FINNGENID", "DATE"], as_index=False).agg({"VALUE": "mean", "SEX":"first", "EVENT_AGE":"first"}, as_index=False)
    data = pd.concat([data, dups]).reset_index(drop=True)

    # Stats
    n_indvs_stats.loc[n_indvs_stats.STEP == "Duplicates","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Duplicates","N_INDVS_NOW"] = len(set(data.FINNGENID))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Duplicates","N_ROWS_REMOVED"] = n_rows-data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Duplicates","N_INDVS_REMOVED"] = n_indv-len(set(data.FINNGENID))

    return(data, n_indvs_stats)
    
"""Removes severe outliers based on the z-scores. Saving plots before and after.
   Relevant for measurements such as eGFR where some extreme measurements are likely errors."""
def remove_severe_value_outliers(data,max_z, n_indvs_stats, res_dir, file_name):
    ### Severe z-score outliers
    data.loc[:,"Z"] = st.zscore(data.VALUE)

    # Make sure not to include single extreme outliers with the value quant. I.e. LDL had one such individual completely skewing the upper box.
    data.loc[:,"VALUE_QUANT"] = pd.qcut(data.loc[np.abs(data.Z) < 100].VALUE + jitter(data.loc[np.abs(data.Z) < 100].VALUE), 100000).apply(lambda x: x.mid)
    data.VALUE_QUANT = data.loc[:,"VALUE_QUANT"].astype("float").round()
    stat_value_quants = data.groupby("VALUE_QUANT").agg({"FINNGENID": lambda x: len(set(x))}).reset_index().rename(columns={"FINNGENID": "N_INDVS"})
    logger.info("Min value quant counts: {}".format(stat_value_quants.N_INDVS.min()))
    # Log with plots
    plt.figure()
    fig, ax = plt.subplots(1,2, figsize=(5,5))
    sns.boxplot(data, y="VALUE_QUANT",ax=ax[0])

    print(data.loc[np.abs(data.Z)<max_z].groupby("VALUE_QUANT").agg({"FINNGENID": lambda x: len(set(x))}).reset_index().rename(columns={"FINNGENID": "N_INDVS"}))
    print(data.loc[np.abs(data.Z)>=max_z].groupby("VALUE_QUANT").agg({"FINNGENID": lambda x: len(set(x))}).reset_index().rename(columns={"FINNGENID": "N_INDVS"}))
    sns.boxplot(data.loc[np.abs(data.Z)<max_z], y="VALUE_QUANT", ax=ax[1])
    ax[0].set_ylabel("Value")
    ax[1].set_ylabel("")
    plt.savefig(res_dir+file_name+"_valuequants.pdf")
    plt.savefig(res_dir+file_name+"_valuequants.png")

    stat_value_quants.to_csv(res_dir+file_name+"_valuequants.csv", sep=",", index=False)
    
    plt.figure()
    fig, ax = plt.subplots(1,2,figsize=(5,5))
    sns.scatterplot(data, x="EVENT_AGE", y="VALUE", hue="Z", ax=ax[0])
    sns.scatterplot(data.loc[np.abs(data.Z)<max_z], x="EVENT_AGE", y="VALUE", hue="Z", ax=ax[1])
    plt.savefig(res_dir+file_name+"_zs.png")
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

""""Removes single value outliers. I.e. individuals with only one measurement that is an outlier.
    For eGFR problems because of wrong recording/conversion of units.
    Only removing values with high z-score>5 and value>500 or low z-score<-4 and value<53.
    Records number of rows and individuals in statistics df n_indvs_stats."""
def remove_single_value_outliers(data, n_indvs_stats):
    # Z-scores for each individuals value distributions
    data.loc[:,"Z_indv"] = data.groupby("FINNGENID").VALUE.transform(lambda x: st.zscore(x))
    # Outliers
    outliers_high = data.loc[data.Z_indv > 2.5].groupby("FINNGENID").filter(lambda x: len(x) == 1).query("Z_indv>5 & VALUE>500")
    outliers_low = data.loc[data.Z_indv < -4].groupby("FINNGENID").filter(lambda x: len(x) == 1).query("VALUE<53")
    outliers = pd.concat([outliers_high, outliers_low])
    # Remove
    data = data.drop(outliers.index)
    # Stats   
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_INDVS_NOW"] =  len(set(data.FINNGENID)) 
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_ROWS_REMOVED"] = outliers.shape[0]

    return(data, n_indvs_stats)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Path to extracted file (step0)", required=True)
    parser.add_argument("--res_dir", help="Path to the results directory (step1)", default="/home/ivm/valid/data/processed_data/step1/")
    parser.add_argument("--lab_name", help="Readable name of the measurement value.", required=True)
    parser.add_argument("--main_unit", help="Unit that will be filtered out as the main unit.", required=True)
    parser.add_argument("--max_z", type=int, help="Maximum z-score among all measurements. [dafult: 10]", default=10)

    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    #### Preparations
    timer = Timer()
    args = get_parser_arguments()
    
    log_dir = args.res_dir + "logs/"
    file_name = args.lab_name + "_" + get_date()
    log_file_name = args.lab_name + "_" + get_datetime()
    make_dir(log_dir)
    make_dir(args.res_dir)
    
    init_logging(log_dir, log_file_name, logger)

    #### Data processing
    ## Raw data
    data = pd.read_csv(args.file_path, sep=",")
    ## Prep
    n_indvs_stats = pd.DataFrame({"STEP": ["Start", "Values", "Unit", "Duplicates", "Outliers", "Outliers_single"]})
    n_indv = len(data.FINNGENID.unique())
    ## Stats
    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_INDVS_NOW"] = n_indv
    ## Cleaning
    data, n_indvs_stats = remove_missing_values(data, n_indvs_stats)
    data, n_indvs_stats = get_main_unit_data(data, n_indvs_stats, args.main_unit)
    data, n_indvs_stats = handle_duplicates(data, n_indvs_stats)
    data, n_indvs_stats = remove_severe_value_outliers(data, args.max_z, n_indvs_stats, log_dir, file_name)
    # For eGFR also removing single individual outliers as the difference in units leads to some clear
    # errors in the data. Such that i.e. one individuals has values around 70 and one around 700 in the middle. 
    # Only removing values with high z-score>5 and value>500 or low z-score<-4 and value<53.
    if args.lab_name == "egfr" or args.lab_name == "krea":
        data, n_indvs_stats = remove_single_value_outliers(data, n_indvs_stats)

    #### Finishing
    ## Saving
    data.to_csv(args.res_dir + file_name + ".csv", sep=",", index=False)
    n_indvs_stats.to_csv(log_dir + file_name + "_counts.csv", sep=",", index=False)
    ## Final logging
    logger.info("Time total: "+timer.get_elapsed())