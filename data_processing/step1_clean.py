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
sns.set_style("whitegrid")
# Statistics/Processing
import scipy.stats as st
from processing_utils import *

def get_main_unit_data(data, n_indvs_stats, main_unit):
    """Get rows where unit is the main unit. Add info to statistics df with STEP 'Unit'."""
    # Stats
    n_indv = len(set(data.FINNGENID))
    n_row_remove = data.loc[data.UNIT != main_unit].shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_ROWS_REMOVED"] = n_row_remove
    # Removing
    data = data.loc[data.UNIT == main_unit].copy()
    # Stats
    n_indv_remove = abs(len(set(data.FINNGENID))-n_indv)
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_INDVS_REMOVED"] = n_indv_remove
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_INDVS_NOW"] = len(set(data.FINNGENID))
       
    return(data, n_indvs_stats)
    
def convert_hba1c_data(data, n_indvs_stats, main_unit):
    """Converting % to mmol/mol, assuming all missing now is mmol/mol and rounding values."""
    data.loc[data.UNIT=="%","VALUE"] = 10.93*data.loc[data.UNIT=="%","VALUE"]-23.50
    data.loc[data.UNIT=="%","UNIT"] = "mmol/mol"
    data.loc[data.UNIT.isnull(),"UNIT"] = "mmol/mol"
    data.VALUE = np.round(data.VALUE)
    
    return(data, n_indvs_stats)

    
def handle_missing_values(data, n_indvs_stats, fill_missing, dummies, dummy_unit):
    """Removes rows without any measurement value. Optionally, fills missing values with dummy
       based on not-missing abnormality.
       Records number of rows and individuals in statistics df n_indvs_stats with STEP 'Values'."""
    # Stats
    n_indv = len(set(data.FINNGENID))
    if fill_missing:
        # Replacing missing values with averages for area
        data.loc[np.logical_and(data.VALUE.isnull(), data.ABNORM == "L"),"UNIT"] = dummy_unit if dummies[0] != -1 else np.nan
        data.loc[np.logical_and(data.VALUE.isnull(), data.ABNORM == "L"),"VALUE"] = dummies[0] if dummies[0] != -1 else np.nan
        data.loc[np.logical_and(data.VALUE.isnull(), data.ABNORM == "N"),"UNIT"] = dummy_unit if dummies[1] != -1 else np.nan
        data.loc[np.logical_and(data.VALUE.isnull(), data.ABNORM == "N"),"VALUE"] = dummies[1] if dummies[1] != -1 else np.nan
        data.loc[np.logical_and(data.VALUE.isnull(), data.ABNORM == "H"),"UNIT"] = dummy_unit if dummies[1] != -1 else np.nan
        data.loc[np.logical_and(data.VALUE.isnull(), data.ABNORM == "H"),"VALUE"] = dummies[2] if dummies[1] != -1 else np.nan
    # Removing
    # (still) missing value
    n_row_remove = data.loc[data.VALUE.isnull()].shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_ROWS_REMOVED"] = n_row_remove
    data = data.loc[data["VALUE"].notnull()].copy()
    # Stats
    n_indv_remove = abs(len(set(data.FINNGENID))-n_indv)
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_INDVS_REMOVED"] = n_indv_remove
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_INDVS_NOW"] = len(set(data.FINNGENID))
       
    return(data, n_indvs_stats)
    

def handle_duplicates(data, n_indvs_stats):
    """Duplicates are measurements at exact same time. Handling so that same value measurements, only
    one is kept. For measurements with different values, removing all measurements at the same
    time where the difference between min and max > 10%IQR of all measurements. As it is not
    possible to know what is causing this extreme difference. 
    Averaging the rest and recording the max-flux value for this test."""
    # Stats
    n_indv = len(set(data.FINNGENID))
    n_rows = data.shape[0]
    ### Duplicates based on exact same time
    dups = data[data.duplicated(subset=["FINNGENID", "DATE"], keep=False)].reset_index(drop=True)
    data = data.drop_duplicates(["FINNGENID", "DATE"], keep=False).reset_index(drop=True).copy()
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
    data = pd.concat([data, dups]).reset_index(drop=True).copy()

    # Stats
    n_indvs_stats.loc[n_indvs_stats.STEP == "Duplicates","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Duplicates","N_INDVS_NOW"] = len(set(data.FINNGENID))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Duplicates","N_ROWS_REMOVED"] = n_rows-data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Duplicates","N_INDVS_REMOVED"] = n_indv-len(set(data.FINNGENID))

    return(data, n_indvs_stats)

def handle_different_units(data, n_indvs_stats, unit_priority=["mmol/mol", "%"]):
    """This approach to duplicate data, handles the duplicataes based on the unit. If there are
    multiple units for the same individual at the same time, the one with the priority unit is kept.
    If there are multiple units with the same priority, the one with the lowest value is kept.
    Removing exact duplicates with the same unit, value, and abnormality. """
    # Stats
    n_indv = len(set(data.FINNGENID))
    n_rows = data.shape[0]

    dups = data[data.duplicated(subset=["FINNGENID", "DATE", "VALUE", "UNIT", "ABNORM"], keep="first")].reset_index(drop=True)
    logger.info("{:,} measurements at the exact same time".format(len(dups)))
        
    # Keeping first of exact duplicates
    data = data.drop_duplicates(subset=["FINNGENID", "DATE", "VALUE", "UNIT", "ABNORM"], keep="first").reset_index(drop=True).copy()
    logging_print("After removing exact duplicates")
    logging_print("{:,} individuals with {:,} rows".format(data.FINNGENID.nunique(), data.shape[0]))

    # Measurements at same time
    dups = data[data.duplicated(subset=["FINNGENID", "DATE"], keep=False)].reset_index(drop=True)
    data = data.drop_duplicates(subset=["FINNGENID", "DATE"], keep=False).reset_index(drop=True).copy()
    logging_print("After temporarily removing date duplicates")
    logging_print("{:,} individuals with {:,} rows".format(data.FINNGENID.nunique(), data.shape[0]))

    # Keeping mmol/mol if available, otherwise %, otherwise no unit made to priority
    dups.UNIT = pd.Categorical(dups.UNIT, unit_priority, ordered=True)
    dups = dups.sort_values(["FINNGENID", "EVENT_AGE", "UNIT", "VALUE"]).groupby(["FINNGENID", "DATE"]).head(1).reset_index(drop=True)
    dups.loc[dups.UNIT.isnull(),"UNIT"] = unit_priority[0]

    ## Adding back top unit dups
    data = pd.concat([data, dups]).reset_index(drop=True)
    logging_print("After adding back units")
    logging_print("{:,} individuals with {:,} rows".format(data.FINNGENID.nunique(), data.shape[0]))

    # Stats
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_INDVS_NOW"] = len(set(data.FINNGENID))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_ROWS_REMOVED"] = n_rows-data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_INDVS_REMOVED"] = n_indv-len(set(data.FINNGENID))

    return(data, n_indvs_stats)

def remove_severe_value_outliers(data, max_z, n_indvs_stats, res_dir, file_name):
    """Removes severe outliers based on the z-scores >= 10. Saving plots before and after."""

    out_dir = res_dir + "plots/"; make_dir(out_dir)
    ### Severe z-score outliers
    data = data.assign(Z=st.zscore(data.VALUE))

    # Make sure not to include single extreme outliers with the value quant. I.e. LDL had one such individual completely skewing the upper box.
    data = data.assign(VALUE_QUANT=pd.qcut(data.loc[np.abs(data.Z) < 100].VALUE + (data.loc[np.abs(data.Z) < 100].VALUE+0.01*(np.random.rand(data.shape[0])-0.5)), 100000).apply(lambda x: x.mid))
    data = data.assign(VALUE_QUANT=data.VALUE_QUANT.astype("float").round())
    stat_value_quants = data.groupby("VALUE_QUANT").agg({"FINNGENID": lambda x: len(set(x))}).reset_index().rename(columns={"FINNGENID": "N_INDVS"}).copy()
    logger.info("Min value quant counts: {}".format(stat_value_quants.N_INDVS.min()))
    # Log with plots
    plt.figure()
    fig, ax = plt.subplots(1,2, figsize=(5,5))
    sns.boxplot(data, y="VALUE_QUANT",ax=ax[0])
    sns.boxplot(data.loc[np.abs(data.Z)<max_z], y="VALUE_QUANT", ax=ax[1])
    ax[0].set_ylabel("Value")
    ax[1].set_ylabel("")
    plt.savefig(out_dir+file_name+"_valuequants.pdf")
    plt.savefig(out_dir+file_name+"_valuequants.png")

    stat_value_quants.to_csv(out_dir+file_name+"_valuequants.csv", sep=",", index=False)
    
    # plt.figure()
    # fig, ax = plt.subplots(1,2,figsize=(5,5))
    # sns.scatterplot(data, x="EVENT_AGE", y="VALUE", hue="Z", ax=ax[0])
    # sns.scatterplot(data.loc[np.abs(data.Z)<max_z], x="EVENT_AGE", y="VALUE", hue="Z", ax=ax[1])
    # plt.savefig(out_dir+file_name+"_zs.png")
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

    return(data, n_indvs_stats)

def remove_known_outliers(data, n_indvs_stats, ref_min, ref_max):
    # Outliers
    n_indv = len(set(data.FINNGENID))
    if not ref_max: ref_max = data.VALUE.max()
    if not ref_min: ref_min = data.VALUE.min()
    n_outliers = data.loc[np.logical_or(data.VALUE<ref_min, data.VALUE>ref_max)].shape[0]
    # Remove
    data = data.loc[np.logical_and(data.VALUE>=ref_min, data.VALUE<=ref_max)].copy()
    # Stats   
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_known","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_known","N_INDVS_NOW"] =  len(set(data.FINNGENID))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_known","N_ROWS_REMOVED"] = n_outliers
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_known","N_INDVS_REMOVED"] = n_indv-len(set(data.FINNGENID))

    return(data, n_indvs_stats)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Path to extracted file (step0)", required=True)
    parser.add_argument("--res_dir", help="Path to the results directory (step1)", default="/home/ivm/valid/data/processed_data/step1/")
    parser.add_argument("--lab_name", help="Readable name of the measurement value.", required=True)
    parser.add_argument("--fill_missing", type=int, help="Whether to remove missing vlaues or set them to a dummies. [deafult remove: 0]", default=0)
    parser.add_argument("--dummies", type=float, nargs="+", help="List with values for low, normal, high missing value abnormal")
    parser.add_argument("--main_unit", help="Unit that will be filtered out as the main unit.")
    parser.add_argument("--max_z", type=int, help="Maximum z-score among all measurements. [dafult: 10]", default=10)
    parser.add_argument("--ref_min", type=float, help="Minimum reasonable value [dafult: None]", default=None)
    parser.add_argument("--ref_max", type=float, help="Minimum reasonable value [dafult: None]", default=None)

    args = parser.parse_args()
    return(args)

def egfr_transform(data):
    data.loc[np.logical_and(data["VALUE"] <= 62, data["SEX"] == "female"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] <= 62, data["SEX"] == "female"), "VALUE"]/61.9 )**(-0.329))*(0.993**data.loc[np.logical_and(data["VALUE"] <= 62, data["SEX"] == "female"), "EVENT_AGE"])*144
    data.loc[np.logical_and(data["VALUE"] > 62, data["SEX"] == "female"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] > 62, data["SEX"] == "female"), "VALUE"]/61.9 )**(-1.209))*(0.993**data.loc[np.logical_and(data["VALUE"] > 62, data["SEX"] == "female"), "EVENT_AGE"])*144
    data.loc[np.logical_and(data["VALUE"] <= 80, data["SEX"] == "male"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] <= 80, data["SEX"] == "male"), "VALUE"]/79.6 )**(-0.411))*(0.993**data.loc[np.logical_and(data["VALUE"] <= 80, data["SEX"] == "male"), "EVENT_AGE"])*141
    data.loc[np.logical_and(data["VALUE"] > 80, data["SEX"] == "male"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] > 80, data["SEX"] == "male"), "VALUE"]/79.6 )**(-1.209))*(0.993**data.loc[np.logical_and(data["VALUE"] > 80, data["SEX"] == "male"), "EVENT_AGE"])*141
    data = data.assign(VALUE = data["VALUE_TRANSFORM"])
    data = data.drop("VALUE_TRANSFORM", axis=1)
    return(data)
    
def custom_abnorm(data, lab_name):
    ### Processing
    # ABNORMity
    if args.lab_name == "tsh":
        data = three_level_abnorm(data)
    if args.lab_name == "hba1c": 
        data = simple_abnorm(data)
    if args.lab_name == "ldl":
        data = simple_abnorm(data)
    if args.lab_name == "egfr" or args.lab_name == "krea": 
        data = egfr_transform(data)
        data = simple_abnorm(data)
    if args.lab_name == "cyst":
        data = simple_abnorm(data)
    if args.lab_name == "gluc" or args.lab_name=="fgluc":
        data = three_level_abnorm(data)  
    else: data = three_level_abnorm(data)
    data = get_abnorm_func_based_on_name(args.lab_name)(data, "VALUE")
    return(data)
    
if __name__ == "__main__":
    #### Preparations
    timer = Timer()
    args = get_parser_arguments()
    
    file_name = args.lab_name + "_d" + str(args.fill_missing) + "_" + get_date() 
    count_dir = args.res_dir + "counts/"
    log_file_name = args.lab_name + "_d" + str(args.fill_missing) + "_" + get_datetime()
    make_dir(args.res_dir); make_dir(count_dir)
    
    init_logging(args.res_dir, log_file_name, logger, args)

    #### Data processing
    # Raw data
    data = pd.read_csv(args.file_path, sep=",")
    # Prep
    n_indvs_stats = pd.DataFrame({"STEP": ["Start", "Dups manual", "Values", "Unit", "Duplicates", "Outliers_known", "Outliers", "Outliers_single"]})
    n_indv = len(set(data.FINNGENID))

    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_INDVS_NOW"] = n_indv
    # Cleaning
    if args.lab_name == "hba1c": 
        data, n_indvs_stats = handle_different_units(data, n_indvs_stats, unit_priority=["mmol/mol", "%"])
        data.loc[data.ABNORM=="A","ABNORM"] = "H"
    data, n_indvs_stats = handle_missing_values(data, n_indvs_stats, args.fill_missing, args.dummies, args.main_unit)

    if args.lab_name != "hba1c":
        data = data.drop(columns={"VALUE", "UNIT", "ABNORM"})
        data = data.rename({"VALUE_FG": "VALUE", "UNIT_FG": "UNIT", "ABNORM_FG": "ABNORM"})
        data, n_indvs_stats = get_main_unit_data(data, n_indvs_stats, args.main_unit)
        data, n_indvs_stats = handle_duplicates(data, n_indvs_stats)
    else:
        data = data.drop(columns={"VALUE_FG", "UNIT_FG", "ABNORM_FG"})
        data, n_indvs_stats = convert_hba1c_data(data, n_indvs_stats, args.main_unit)

    data, n_indvs_stats = remove_known_outliers(data, n_indvs_stats, args.ref_min, args.ref_max)
    print(n_indvs_stats)
    data, n_indvs_stats = remove_severe_value_outliers(data, args.max_z, n_indvs_stats, args.res_dir, file_name)

    if args.lab_name == "egfr" or args.lab_name == "krea":
        data, n_indvs_stats = remove_single_value_outliers(data, n_indvs_stats)

    #### Finishing
    data = custom_abnorm(data, args.lab_name)
    logging.info("Min abnorm " + str(data.loc[data.ABNORM_CUSTOM == 1.0].VALUE.min()) + " max abnorm " + str(data.loc[data.ABNORM_CUSTOM == 1.0].VALUE.max()))
    
    # Saving
    data = data.drop(columns=["Z", "VALUE_QUANT"])
    data.to_csv(args.res_dir + file_name + ".csv", sep=",", index=False)
    n_indvs_stats.to_csv(count_dir + file_name + "_counts.csv", sep=",", index=False)
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
