# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer, get_date, get_datetime, make_dir, init_logging, logging_print, read_file
from processing_utils import three_level_abnorm, simple_abnorm, egfr_transform, get_abnorm_func_based_on_name
# Standard stuff
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
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
import warnings

def remove_single_value_outliers(data: pl.DataFrame, 
                                 n_indvs_stats: pd.DataFrame) -> tuple[pl.DataFrame, pd.DataFrame]:
    """This is geared towards eGFR"""
    # Z-scores for each individuals value distributions
    data = (data.with_columns(
                    ((pl.col("VALUE") - pl.col("VALUE").mean().over("FINNGENID")) /
                     pl.col("VALUE").std().over("FINNGENID")).alias("Z_indv")))
    # with warnings.catch_warnings(action="ignore"):
    #     z_scores = (data.group_by("FINNGENID", maintain_order=True)
    #                 .agg(pl.col("VALUE").map_elements(lambda group_values: st.zscore(group_values.to_numpy())).alias("Z_indv").explode()))
    #     print(z_scores)
    #data = data.join(z_scores.select(["FINNGENID", "Z_indv"]), on="FINNGENID")
    # Outliers
    print(data.sort(["FINNGENID", "DATE"]))
    outliers_high = (data.filter(pl.col("Z_indv")>2.5)).group_by("FINNGENID").agg(pl.len().alias("N_ROW"))
    outliers_high = (data.join(outliers_high, on="FINNGENID")# individuals with single severe outliers 
                         .filter((pl.col("N_ROW")==1) & (pl.col("Z_indv")>5) & (pl.col("VALUE")>500)))
                         
    outliers_low = (data.filter(pl.col("Z_indv")<-4)).group_by("FINNGENID").agg(pl.len().alias("N_ROW"))
    outliers_low = (data.join(outliers_low, on="FINNGENID")
                        .filter((pl.col("N_ROW")==1) & (pl.col("VALUE")>53)))
    outliers = pl.concat([outliers_high, outliers_low])
    print(outliers)
    # Remove
    data = data.filter(~pl.col("FINNGENID").is_in(outliers["FINNGENID"]))
    data = data.drop("Z_indv")

    # Stats   
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_INDVS_NOW"] =  len(set(data["FINNGENID"])) 
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_ROWS_REMOVED"] = outliers.height

    return(data, n_indvs_stats)

def remove_severe_value_outliers(data: pl.DataFrame, 
                                 n_indvs_stats: pd.DataFrame,
                                 max_z: float, 
                                 res_dir: str, 
                                 file_name: str, 
                                 plot=True) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Removes severe outliers based on the z-scores >= max_z. Saving plots before and after."""

    out_dir = res_dir + "plots/"; make_dir(out_dir)
    ### Severe z-score outliers
    data = data.with_columns(
        pl.lit(st.zscore(data["VALUE"].to_numpy())).alias("Z")
    )
    # Make sure not to include single extreme outliers with the value quant. I.e. LDL had one such individual completely skewing the upper box.
    # Log with plots
    if plot:
        print("start")
        plt.figure()
        fig, ax = plt.subplots(1,2,figsize=(5,5))
        sns.scatterplot(data, x="EVENT_AGE", y="VALUE", ax=ax[0], hue="Z")
        sns.scatterplot(data.filter(pl.col("Z").abs()<max_z), x="EVENT_AGE", y="VALUE", ax=ax[1], hue="Z")
        plt.savefig(out_dir+file_name+"_zs.png")
        print("end")
    # Stats
    n_indv = len(set(data["FINNGENID"]))
    n_row_remove = data.filter(pl.col("Z").abs()>=max_z).height
    logger.info("Max z of {} removed values greater than {} and lower than {}".format(max_z, 
                                                                                      data.filter(pl.col("Z")>=max_z).select(pl.col("VALUE").min()).to_numpy()[0][0], 
                                                                                      data.filter(pl.col("Z")<=-max_z).select(pl.col("VALUE").max()).to_numpy()[0][0]))
    # Removing
    data = data.filter(pl.col("Z").abs()<max_z)
    # Stats
    n_indv_remove = n_indv - len(set(data["FINNGENID"]))
    n_indv = len(set(data["FINNGENID"]))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers","N_INDVS_NOW"] = n_indv 
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers","N_INDVS_REMOVED"] = n_indv_remove
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers","N_ROWS_REMOVED"] = n_row_remove

    return(data, n_indvs_stats)

def remove_known_outliers(data: pl.DataFrame, 
                          n_indvs_stats: pd.DataFrame, 
                          ref_min: float, 
                          ref_max: float) -> tuple[pl.DataFrame, pd.DataFrame]:
    # Outliers
    n_indv = len(set(data["FINNGENID"]))
    if not ref_max: ref_max = data.select(pl.col("VALUE").max()).to_numpy()[0][0]
    if not ref_min: ref_min = data.select(pl.col("VALUE").min()).to_numpy()[0][0]
    n_outliers = data.filter((pl.col("VALUE")<ref_min) | (pl.col("VALUE")>ref_max)).height
    data = data.filter((pl.col("VALUE")>=ref_min) & (pl.col("VALUE")<=ref_max))
    # Stats   
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_known","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_known","N_INDVS_NOW"] =  len(set(data["FINNGENID"]))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_known","N_ROWS_REMOVED"] = n_outliers
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_known","N_INDVS_REMOVED"] = n_indv-len(set(data["FINNGENID"]))

    return(data, n_indvs_stats)

def custom_abnorm(data: pl.DataFrame, 
                  lab_name: str) -> pl.DataFrame:
    """Chooses abnormality function based on lab name and applies it to data"""
    if lab_name == "tsh":
        data = three_level_abnorm(data)
    if lab_name == "hba1c": 
        data = simple_abnorm(data)
    if lab_name == "ldl":
        data = simple_abnorm(data)
    if lab_name == "egfr" or lab_name == "krea": 
        data = egfr_transform(data)
        data = simple_abnorm(data)
    if lab_name == "cyst":
        data = simple_abnorm(data)
    if lab_name == "gluc" or lab_name=="fgluc":
        data = three_level_abnorm(data)  
    else: data = three_level_abnorm(data)
    
    data = get_abnorm_func_based_on_name(lab_name)(data, "VALUE")

    return(data)


def convert_hba1c_data(data, n_indvs_stats, main_unit):
    """Converting % to mmol/mol, assuming all missing now is mmol/mol and rounding values.
       Wrong units are very likely to get removed in a later step."""
    data = data.with_columns([pl.when(pl.col("UNIT") == "%")
                               .then(10.93*pl.col("VALUE")-23.50)
                               .otherwise(pl.col("VALUE"))
                               .alias("VALUE"),
                              pl.lit("mmol/mol").alias("UNIT")])
    data = data.with_columns(pl.col("VALUE").round().alias("VALUE"))
    
    return(data, n_indvs_stats)

def get_main_unit_data(data: pl.DataFrame, 
                       n_indvs_stats: pd.DataFrame, 
                       main_unit: str) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Get rows where unit is the main unit. Add info to statistics df with STEP 'Unit'."""
    # Stats
    n_indv = len(set(data["FINNGENID"]))
    n_row_remove = data.filter(pl.col("UNIT") != main_unit).height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_ROWS_REMOVED"] = n_row_remove
    # Removing
    data = data.filter(pl.col("UNIT") == main_unit)
    # Stats
    n_indv_remove = abs(len(set(data["FINNGENID"]))-n_indv)
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_INDVS_REMOVED"] = n_indv_remove
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Unit","N_INDVS_NOW"] = len(set(data["FINNGENID"]))
       
    return(data, n_indvs_stats)
    
def handle_missing_values(data: pl.DataFrame, 
                          n_indvs_stats: pd.DataFrame, 
                          fill_missing: bool, 
                          dummies: list[float], 
                          dummy_unit: str) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Removes rows without any measurement value. Optionally, fills missing values with dummy
       based on not-missing abnormality.
       Records number of rows and individuals in statistics df n_indvs_stats with STEP 'Values'."""
    # Stats
    n_indv = len(set(data["FINNGENID"]))
    if fill_missing:
        logging_print(f"Number of missing rows with abnorm being filled: {data.filter((pl.col("VALUE").is_null()) & (pl.col("ABNORM").is_not_null())).height}")
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #                 Replacing missing                                       #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #          
        data = data.with_columns(pl.when((pl.col("VALUE").is_null()) & (pl.col("ABNORM") == "L"))
                                   .then(pl.lit(dummy_unit) if dummies[0] != -1 else pl.lit(None))
                                   .otherwise(pl.col("UNIT"))
                                   .alias("UNIT"))
        data = data.with_columns(pl.when((pl.col("VALUE").is_null()) & (pl.col("ABNORM") == "L"))
                                   .then(dummies[0] if dummies[0] != -1 else None)
                                   .otherwise(pl.col("VALUE"))
                                   .alias("VALUE"))
        data = data.with_columns(pl.when((pl.col("VALUE").is_null()) & (pl.col("ABNORM") == "N"))
                                   .then(pl.lit(dummy_unit) if dummies[1] != -1 else pl.lit(None))
                                   .otherwise(pl.col("UNIT"))
                                   .alias("UNIT"))
        data = data.with_columns(pl.when((pl.col("VALUE").is_null()) & (pl.col("ABNORM") == "N"))
                                   .then(dummies[1] if dummies[1] != -1 else None)
                                   .otherwise(pl.col("VALUE"))
                                   .alias("VALUE"))
        data = data.with_columns(pl.when((pl.col("VALUE").is_null()) & (pl.col("ABNORM") == "H"))
                                   .then(pl.lit(dummy_unit) if dummies[2] != -1 else pl.lit(None))
                                   .otherwise(pl.col("UNIT"))
                                   .alias("UNIT"))
        data = data.with_columns(pl.when((pl.col("VALUE").is_null()) & (pl.col("ABNORM") == "H"))
                                   .then(dummies[2] if dummies[2] != -1 else None)
                                   .otherwise(pl.col("VALUE"))
                                   .alias("VALUE"))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Removing (still) missing values                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
    n_row_remove = data.filter(pl.col("VALUE").is_null()).height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_ROWS_REMOVED"] = n_row_remove
    data = data.filter(pl.col("VALUE").is_not_null())

    # Stats
    n_indv_remove = abs(len(set(data["FINNGENID"]))-n_indv)
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_INDVS_REMOVED"] = n_indv_remove
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Values","N_INDVS_NOW"] = len(set(data["FINNGENID"]))
       
    return(data, n_indvs_stats)
    
def handle_different_units(data: pl.DataFrame, 
                           n_indvs_stats: pd.DataFrame, 
                           unit_priority=["mmol/mol", "%"]) -> tuple[pl.DataFrame, pd.DataFrame]:
    """This approach to duplicate data, handles the duplicataes based on the unit. If there are
    multiple units for the same individual at the same time, the one with the priority unit is kept.
    If there are multiple units with the same priority, the one with the lowest value is kept.
    Removing exact duplicates with the same unit, value, and abnormality. """
    # Stats
    n_indv = len(set(data["FINNGENID"]))
    n_rows = data.height

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Exact duplicates                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    dups = data.filter(data.select(["FINNGENID", "DATE", "VALUE", "UNIT"]).is_duplicated())
    logger.info("{:,} measurements at the exact same time".format(len(dups)))
        
    # Keeping first of exact duplicates
    data = data.unique(subset=["FINNGENID", "DATE", "VALUE", "UNIT"], keep="first")
    logging_print("After removing exact duplicates")

    logging_print("{:,} individuals with {:,} rows".format(len(data["FINNGENID"].unique()), data.height))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Time duplicates                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    dups = data.filter(data.select(["FINNGENID", "DATE"]).is_duplicated())
    data = data.unique(subset=["FINNGENID", "DATE"], keep="none")
    logging_print("After temporarily removing date duplicates")
    logging_print("{:,} individuals with {:,} rows".format(len(data["FINNGENID"].unique()), data.height))

    # Keeping mmol/mol if available, otherwise %, otherwise no unit made to priority
    dups = dups.sort(["FINNGENID", "EVENT_AGE", "UNIT", "VALUE"], nulls_last=True).group_by(["FINNGENID", "DATE"]).first()
    dups = dups.with_columns(pl.when(pl.col("UNIT").is_null())
                               .then(pl.lit(unit_priority[0]))
                               .otherwise(pl.col("UNIT"))
                               .alias("UNIT"))
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Adding back top dups and stats                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    data = pl.concat([data, dups.select(data.columns)])
    logging_print("After adding back units")
    logging_print("{:,} individuals with {:,} rows".format(len(data["FINNGENID"].unique()), data.height))

    # Stats
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_INDVS_NOW"] = len(set(data["FINNGENID"]))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_ROWS_REMOVED"] = n_rows-data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_INDVS_REMOVED"] = n_indv-len(set(data["FINNGENID"]))

    return(data, n_indvs_stats)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Info for files
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--file_path", help="Path to extracted file (step0)", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)

    # Processing settings
    parser.add_argument("--fill_missing", type=int, help="Whether to remove missing vlaues or set them to a dummies. [deafult remove: 0]", default=0)
    parser.add_argument("--dummies", type=float, nargs="+", help="List with values for low, normal, high missing value abnormal")
    parser.add_argument("--priority_units", type=str, default=["mmol/mol", "%"], nargs="+", help="List with values for low, normal, high missing value abnormal")
    parser.add_argument("--main_unit", help="Unit that will be filtered out as the main unit.")
    parser.add_argument("--max_z", type=int, help="Maximum z-score among all measurements. [dafult: 10]", default=10)
    parser.add_argument("--ref_min", type=float, help="Minimum reasonable value [dafult: None]", default=None)
    parser.add_argument("--ref_max", type=float, help="Maximum reasonable value [dafult: None]", default=None)
    parser.add_argument("--plot", type=int, help="Whether to create plots for quality control (time-consuming).", default=1)

    args = parser.parse_args()
    return(args)

def replace_missing_with_extracted(data, n_indvs_stats):
    data.group_by(["FINNGENID", "DATE"])
    
    return(data, n_indvs_stats)
if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    # File names and directories
    file_name = args.lab_name + "_d" + str(args.fill_missing) + "_" + get_date() 
    count_dir = args.res_dir + "counts/"
    log_file_name = args.lab_name + "_d" + str(args.fill_missing) + "_" + get_datetime()
    make_dir(args.res_dir); make_dir(count_dir)
    # Logging
    init_logging(args.res_dir, log_file_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Getting data                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #       
    data = read_file(args.file_path)
    # Prep
    n_indvs_stats = pd.DataFrame({"STEP": ["Start", "Dups manual", "Values", "Outliers_known", "Outliers", "Outliers_single"]})
    n_indv = len(set(data["FINNGENID"]))

    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_INDVS_NOW"] = n_indv

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Processing                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    if args.lab_name != "hba1c":
        data = data.drop(["VALUE", "UNIT"]).rename({"VALUE_FG": "VALUE", "UNIT_FG": "UNIT"})
    if args.lab_name == "hba1c": 
        data = data.drop(["VALUE_FG", "UNIT_FG", "ABNORM_FG"])
        data = data.with_columns(pl.when(pl.col("ABNORM")=="A")
                                   .then(pl.lit("H"))
                                   .otherwise(pl.col("ABNORM"))
                                   .alias("ABNORM"))
    data, n_indvs_stats = handle_different_units(data, n_indvs_stats, unit_priority=args.priority_units)
    data, n_indvs_stats = handle_missing_values(data, n_indvs_stats, args.fill_missing, args.dummies, args.main_unit)
    if "ABNORM_FG" in data.columns:
        data = data.drop("ABNORM").rename({"ABNORM_FG": "ABNORM"})
    if args.lab_name != "hba1c":
        data, n_indvs_stats = get_main_unit_data(data, n_indvs_stats, args.main_unit)
    else:
        data, n_indvs_stats = convert_hba1c_data(data, n_indvs_stats, args.main_unit)

    data, n_indvs_stats = remove_known_outliers(data, n_indvs_stats, args.ref_min, args.ref_max)
    print(n_indvs_stats)
    data, n_indvs_stats = remove_severe_value_outliers(data, n_indvs_stats, args.max_z, args.res_dir, file_name, args.plot)
    print(n_indvs_stats)

    if args.lab_name == "egfr" or args.lab_name == "krea":
        data, n_indvs_stats = remove_single_value_outliers(data, n_indvs_stats)
    print(n_indvs_stats)

    #### Finishing
    data = custom_abnorm(data, args.lab_name)
    logging.info("Min abnorm " + str(data.filter(pl.col("ABNORM_CUSTOM")>0.0).select(pl.col("VALUE").min()).to_numpy()[0][0]) + " max abnorm " + str(data.filter(pl.col("ABNORM_CUSTOM")>0.0).select(pl.col("VALUE").max()).to_numpy()[0][0]))
    
    # Saving
    data = data.drop(["Z"])
    data.write_parquet(args.res_dir + file_name + ".parquet")
    n_indvs_stats.to_csv(count_dir + file_name + "_counts.csv", sep=",", index=False)
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
