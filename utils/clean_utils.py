# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from processing_utils import get_abnorm_func_based_on_name, three_level_abnorm, simple_abnorm
# Standard stuff
import pandas as pd
import polars as pl
from general_utils import make_dir, logging_print
import scipy.stats as st
# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
sns.set_style("whitegrid")
# Logging and input
import logging
from typing import tuple

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
    outliers_high = (data.filter(pl.col("Z_indv")>2.5)).group_by("FINNGENID").agg(pl.len().alias("N_ROW"))
    outliers_high = (data.join(outliers_high, on="FINNGENID")# individuals with single severe outliers 
                         .filter((pl.col("N_ROW")==1) & (pl.col("Z_indv")>5) & (pl.col("VALUE")>500)))
                         
    outliers_low = (data.filter(pl.col("Z_indv")<-4)
                        .group_by("FINNGENID")
                        .agg(pl.len().alias("N_ROW")))
    outliers_low = (data.join(outliers_low, on="FINNGENID")
                        .filter((pl.col("N_ROW")==1) & (pl.col("VALUE")<53)))
    outliers = pl.concat([outliers_high, outliers_low])
    print(outliers)
    # Remove
    data = data.filter(~pl.col("FINNGENID").is_in(outliers["FINNGENID"]))
    data = data.drop("Z_indv")

    # Stats   
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_INDVS_NOW"] =  len(set(data["FINNGENID"])) 
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_ROWS_REMOVED"] = outliers.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Outliers_single","N_INDVS_REMOVED"] = 0

    return(data, n_indvs_stats)

def remove_severe_value_outliers(data: pl.DataFrame, 
                                 n_indvs_stats: pd.DataFrame,
                                 max_z: float, 
                                 res_dir: str, 
                                 file_name: str, 
                                 logger: logging.Logger,
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
        if data.height > 1000000:
            subset_data = data.sample(1000000)
        else:
            subset_data = data
        print("start")
        plt.figure()
        fig, ax = plt.subplots(1,2,figsize=(5,5))
        sns.scatterplot(subset_data, x="EVENT_AGE", y="VALUE", ax=ax[0], hue="Z")
        sns.scatterplot(subset_data.filter(pl.col("Z").abs()<max_z), x="EVENT_AGE", y="VALUE", ax=ax[1], hue="Z")
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

def convert_hba1c_data(data: pl.DataFrame, 
                       n_indvs_stats: pd.DataFrame) -> tuple[pl.DataFrame, pd.DataFrame]:
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
        logging_print(f"Number of missing rows with abnorm being filled: {data.filter((pl.col('VALUE').is_null()) & (pl.col('ABNORM').is_not_null())).height}")
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
        data = data.with_columns(pl.when((pl.col("VALUE").is_null()) & (pl.col("ABNORM") == "LL"))
                                   .then(dummies[3] if dummies[3] != -1 else None)
                                   .otherwise(pl.col("VALUE"))
                                   .alias("VALUE"))
        data = data.with_columns(pl.when((pl.col("VALUE").is_null()) & (pl.col("ABNORM") == "HH"))
                                   .then(dummies[4] if dummies[4] != -1 else None)
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

def handle_same_day_duplicates(data: pl.DataFrame,
                               n_indvs_stats: pd.DataFrame,
                               keep_last_of_day: int=0) -> pl.DataFrame:
    """Mean of duplicates at the same time. K"""
    # Stats
    n_indv = len(set(data["FINNGENID"]))
    n_rows = data.height

    # truncating datetime to day
    if data.schema["DATE"]==pl.Datetime:
        data = data.with_columns(pl.col.DATE.dt.date().alias("DATE_SHORT"))
    else:
        print(data)
        data = data.with_columns(
                    pl.col.DATE.cast(pl.Utf8).str.to_date("%Y-%m-%d %H:%M:%S", strict=False).dt.date()
                    .fill_null(pl.col.DATE.cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False))
                    .alias("DATE_SHORT")
        )
        print(data)

    dups = data.filter(data.select(["FINNGENID", "DATE_SHORT"]).is_duplicated())
    logging_print("{:,} measurements on the same day".format(len(dups)))
    dups = data.filter(data.select(["FINNGENID", "DATE"]).is_duplicated())
    logging_print("{:,} measurements at the exact same time with different values".format(len(dups)))

    if keep_last_of_day == 0:
        # Averaging
        if "ABNORM" in data.columns:  
            if "UNIT" in data.columns:
                data = (data.group_by(["FINNGENID", "DATE_SHORT", "EVENT_AGE", "SEX", "ABNORM", "UNIT"])
                        .agg(pl.col("VALUE").mean().alias("VALUE")))
            else:
                data = (data.group_by(["FINNGENID", "DATE_SHORT", "EVENT_AGE", "SEX", "ABNORM"])
                    .agg(pl.col("VALUE").mean().alias("VALUE")))
        else:
            data = (data.group_by(["FINNGENID", "DATE_SHORT", "EVENT_AGE", "SEX"])
                    .agg(pl.col("VALUE").mean().alias("VALUE")))
        data = data.rename({"DATE_SHORT": "DATE"})
    else:
        print(data)
        # Taking last of days
        data = (data
                  .sort("FINNGENID", "DATE")
                  .filter((pl.col("DATE")==pl.col("DATE").last()).over("FINNGENID", "DATE_SHORT"))
                  .drop("DATE").rename({"DATE_SHORT": "DATE"})
               )
    logging_print("After removing exact duplicates")
    logging_print("{:,} individuals with {:,} rows".format(len(data["FINNGENID"].unique()), data.height))
    print(data)

    # Stats
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups mean","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups mean","N_INDVS_NOW"] = len(set(data["FINNGENID"]))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups mean","N_ROWS_REMOVED"] = n_rows-data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups mean","N_INDVS_REMOVED"] = n_indv-len(set(data["FINNGENID"]))

    return(data, n_indvs_stats)
 
def handle_exact_duplicates(data: pl.DataFrame,
                            n_indvs_stats: pd.DataFrame) -> pl.DataFrame:
    """Removes exact duplicates based on FINNGENID, DATE, VALUE and UNIT (if unit avaliable)."""
    # Stats
    n_indv = len(set(data["FINNGENID"]))
    n_rows = data.height

    if "UNIT" in data.columns:
        dups = data.filter(data.select(["FINNGENID", "DATE", "VALUE", "UNIT"]).is_duplicated())
    else:
        dups = data.filter(data.select(["FINNGENID", "DATE", "VALUE"]).is_duplicated())

    logging_print("{:,} measurements at the exact same time".format(len(dups)))
        
    # Keeping first of exact duplicates
    if "UNIT" in data.columns:
        data = data.unique(subset=["FINNGENID", "DATE", "VALUE", "UNIT"], keep="first")
    else:
        data = data.unique(subset=["FINNGENID", "DATE", "VALUE"], keep="first")
    logging_print("After removing exact duplicates")
    logging_print("{:,} individuals with {:,} rows".format(len(data["FINNGENID"].unique()), data.height))

    # Stats
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups exact","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups exact","N_INDVS_NOW"] = len(set(data["FINNGENID"]))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups exact","N_ROWS_REMOVED"] = n_rows-data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups exact","N_INDVS_REMOVED"] = n_indv-len(set(data["FINNGENID"]))

    return(data, n_indvs_stats)

def handle_multi_unit_duplicates(data: pl.DataFrame,
                                 unit_priority) -> pl.DataFrame:
    """This approach to duplicate data, handles the duplicataes based on the time. If there are
    multiple measurements for the same individual at the same time, the one with the lowest value is kept.
    Removing exact duplicates with the same unit, value, and abnormality. """
    dups = data.filter(data.select(["FINNGENID", "DATE"]).is_duplicated())
    data = data.unique(subset=["FINNGENID", "DATE"], keep="none")
    logging_print("After temporarily removing date duplicates")
    logging_print("{:,} individuals with {:,} rows".format(len(data["FINNGENID"].unique()), data.height))
    # Keeping mmol/mol if available, otherwise %, otherwise no unit made to priority
    dups = (dups
            .sort(["FINNGENID", "EVENT_AGE", "UNIT", "VALUE"], nulls_last=True)
            .group_by(["FINNGENID", "DATE"])
            .first())
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

    
    return(data)

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

    data, n_indvs_stats = handle_exact_duplicates(data, n_indvs_stats)
    data = handle_multi_unit_duplicates(data, unit_priority)

    # Stats
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_INDVS_NOW"] = len(set(data["FINNGENID"]))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_ROWS_REMOVED"] = n_rows-data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Dups manual","N_INDVS_REMOVED"] = n_indv-len(set(data["FINNGENID"]))

    return(data, n_indvs_stats)
