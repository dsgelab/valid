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


"""Get medication and diagnosis information based on the diagnosis and medication regex given."""
def get_diag_med_data(diag_regex,
                      med_regex,
                      table = f"`finngen-production-library.sandbox_tools_r12.finngen_r12_service_sector_detailed_longitudinal_v1`"):
    timer = Timer()
    print(diag_regex)
    print(med_regex)
    if med_regex != "":
        query = f"""SELECT FINNGENID, SOURCE, EVENT_AGE, APPROX_EVENT_DAY, CODE1, CODE2
                         FROM {table}
                         WHERE (SOURCE IN ("INPAT", "OUTPAT", "PRIM_OUT", "REIMB") AND REGEXP_CONTAINS(CODE2, "{diag_regex}")) OR 
                               (SOURCE IN ("INPAT", "OUTPAT", "PRIM_OUT") AND REGEXP_CONTAINS(CODE1, "{diag_regex}")) OR 
                               (SOURCE = "PURCH" AND REGEXP_CONTAINS(CODE1, "{med_regex}"))"""
    else:
        query = f"""SELECT FINNGENID, SOURCE, EVENT_AGE, APPROX_EVENT_DAY, CODE1, CODE2
                         FROM {table}
                         WHERE (SOURCE IN ("INPAT", "OUTPAT", "PRIM_OUT", "REIMB") AND REGEXP_CONTAINS(CODE2, "{diag_regex}")) OR 
                               (SOURCE IN ("INPAT", "OUTPAT", "PRIM_OUT") AND REGEXP_CONTAINS(CODE1, "{diag_regex}"))"""
    diags = query_to_df(query)
    # CODE1 is diagnosis and CODE2 symptom
    diags = pd.melt(diags, id_vars=["FINNGENID", "EVENT_AGE", "APPROX_EVENT_DAY"])
    diags = diags.drop(columns=["variable"]).drop_duplicates()
    diags = diags.rename(columns={"value":"CODE"})
    # Other codes are numbers
    diags = diags.loc[diags.CODE.notnull()]
    diags = diags.loc[diags.CODE.str.contains("^[A-Z][0-9]")]
    print(diags)
    logger.info("Time import: "+timer.get_elapsed())
    return(diags)

"""First diagnosis either due to code or medication purchase."""
def add_first_diag_info(data, diags, diag_regex):
    diags = diags.loc[diags.CODE.str.contains(diag_regex)]
    first_diag = diags.sort_values(["FINNGENID", "APPROX_EVENT_DAY"], ascending=True).groupby("FINNGENID").head(1)
    first_diag = first_diag.rename(columns={"APPROX_EVENT_DAY": "DIAG_DATE", "CODE":"DIAG"})
    data = pd.merge(data, first_diag[["FINNGENID", "DIAG_DATE", "DIAG"]], how="left", on="FINNGENID")

    return(data)
    
"""First medication purchase."""
def add_first_med_info(data, diags, med_regex):
    diags = diags.loc[diags.CODE.str.contains(med_regex)]
    first_meds = diags.sort_values(["FINNGENID", "APPROX_EVENT_DAY"], ascending=True).groupby("FINNGENID").head(1)
    first_meds = first_meds.rename(columns={"APPROX_EVENT_DAY":"MED_DATE", "CODE":"MED"})
    data = pd.merge(data, first_meds[["FINNGENID", "MED_DATE", "MED"]], how="left", on="FINNGENID")

    return(data)

"""Adds a column with three values: Last measurement (1), First measurement (0) and second to last measurement (2).
   When only one measurement -> 1. Only two measurements -> 1 and 0. """
def add_edge_info(data):
    last_data = data.sort_values(["FINNGENID", "DATE"], ascending=False).groupby("FINNGENID").head(1)[["FINNGENID", "DATE"]]
    last_data["EDGE"] = 1
    three_val_data = data.groupby("FINNGENID").filter(lambda x: len(x) > 2)
    scnd_last_data = three_val_data.sort_values(["FINNGENID", "DATE"], ascending=False).groupby("FINNGENID").head(2).groupby("FINNGENID").tail(1)[["FINNGENID", "DATE"]]
    scnd_last_data["EDGE"] = 2
    two_val_data = data.groupby("FINNGENID").filter(lambda x: len(x) > 1)
    first_data = two_val_data.sort_values(["FINNGENID", "DATE"], ascending=True).groupby("FINNGENID").head(1)[["FINNGENID", "DATE"]]
    first_data["EDGE"] = 0
    edge_data = pd.concat([last_data, scnd_last_data, first_data])
    data = pd.merge(data, edge_data, on=["FINNGENID", "DATE"], how="left").sort_values(["FINNGENID", "DATE"])
    return(data)
    
"""TODO: Problem flexible min age"""
def add_age_excl_col(data, n_indvs_stats, min_age=18, max_age=70):
    too_young = set(data.groupby("FINNGENID").agg({"EVENT_AGE":"max"}).query("EVENT_AGE<18").index)
    too_old = set(data.loc[data["EVENT_AGE"] > max_age].FINNGENID)
    age_exclusions = too_young | too_old
    data.loc[data.FINNGENID.isin(age_exclusions),"AGE_EXCLUDE"] = 1
    data.loc[~data.FINNGENID.isin(age_exclusions),"AGE_EXCLUDE"] = 0

    n_indvs_stats.loc[n_indvs_stats.STEP == "Age","N_ROWS_NOW"] = data.query("AGE_EXCLUDE==0").shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Age","N_INDVS_NOW"] = len(set(data.query("AGE_EXCLUDE==0").FINNGENID))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Age","N_ROWS_REMOVED"] = data.query("AGE_EXCLUDE==1").shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Age","N_INDVS_REMOVED"] = len(set(data.query("AGE_EXCLUDE==1").FINNGENID))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Age min two","N_ROWS_NOW"] = data.query("N_YEAR > 1 & AGE_EXCLUDE==0").shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Age min two","N_INDVS_NOW"] = len(set(data.query("N_YEAR > 1 & AGE_EXCLUDE==0").FINNGENID))
    n_indvs_stats.loc[n_indvs_stats.STEP == "Age min two","N_ROWS_REMOVED"] = data.query("N_YEAR > 1 & AGE_EXCLUDE==1").shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Age min two","N_INDVS_REMOVED"] = len(set(data.query("N_YEAR > 1 & AGE_EXCLUDE==1").FINNGENID))
    return(data, n_indvs_stats)


def add_hbb_col(data, n_indvs_stats):
    table = f"`finngen-production-library.sandbox_tools_r12.minimum_extended_r12_v1`"
    query = f""" SELECT FINNGENID
                     FROM {table}
                     WHERE COHORT = "HELSINKI BIOBANK"
             	 """
    hbb_ids = query_to_df(query).FINNGENID.values
    data.loc[data.FINNGENID.isin(hbb_ids),"HBB"] = 1
    data.loc[~data.FINNGENID.isin(hbb_ids),"HBB"] = 0
    
    n_indvs_stats.loc[n_indvs_stats.STEP == "HBB","N_ROWS_NOW"] = data.query("HBB==1").shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "HBB","N_INDVS_NOW"] = len(set(data.query("HBB==1").FINNGENID))

    n_indvs_stats.loc[n_indvs_stats.STEP == "HBB age","N_ROWS_NOW"] = data.query("AGE_EXCLUDE==0 & HBB==1").shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "HBB age","N_INDVS_NOW"] = len(set(data.query("AGE_EXCLUDE==0 & HBB==1").FINNGENID))

    n_indvs_stats.loc[n_indvs_stats.STEP == "HBB age min two","N_ROWS_NOW"] = data.query("N_YEAR > 1 & AGE_EXCLUDE==0 & HBB==1").shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "HBB age min two","N_INDVS_NOW"] = len(set(data.query("N_YEAR > 1 &AGE_EXCLUDE==0 & HBB==1").FINNGENID))

    return(data, n_indvs_stats)

def add_measure_counts(data):
    data.DATE = data.DATE.astype("datetime64[ns]")
    n_measures = data.groupby("FINNGENID").agg({"DATE": [("FIRST_LAST", np.ptp), ("N_YEAR", lambda x: len(set(x.dt.year)))], "VALUE": len}).reset_index()
    n_measures.columns = ["FINNGENID", "FIRST_LAST", "N_YEAR", "N_MEASURE"]
    data = pd.merge(data, n_measures, on="FINNGENID", how="left")
    data.FIRST_LAST = data.FIRST_LAST.dt.days
    return(data)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Path to extracted file (step1)", required=True)
    parser.add_argument("--res_dir", help="Path to the results directory (step2)", default="/home/ivm/valid/data/processed_data/step2/")
    parser.add_argument("--lab_name", help="Readable name of the measurement value.", required=True)
    parser.add_argument("--diag_regex", type=str, help="Regex for selecting Code1 and 2 from the service sector detailed longitudinal data.", required=True)
    parser.add_argument("--med_regex", type=str, help="Regex for selecting medication purchases.", required=False, default="")
    args = parser.parse_args()
    return(args)

def init_logging(log_dir, log_file_name, date_time):
    logging.basicConfig(filename=log_dir+log_file_name+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.info("Time: " + date_time + " Args: --" + ' --'.join(f'{k}={v}' for k, v in vars(args).items()))
    
if __name__ == "__main__":
    #### Preparations
    timer = Timer()
    args = get_parser_arguments()
    
    log_dir = args.res_dir + "logs/"
    file_name = args.lab_name + "_" + get_date()
    log_file_name = "meta_" + args.lab_name + "_" + get_datetime()
    make_dir(log_dir)
    make_dir(args.res_dir)
    
    init_logging(log_dir, log_file_name, logger)

    #### Data processing
    # Raw data
    data = pd.read_csv(args.file_path, sep=",")
    # Extra columns
    n_indvs_stats = pd.DataFrame({"STEP": ["Age", "Age min two", "HBB", "HBB age", "HBB age min two"]})

    data = add_measure_counts(data)
    data, n_indvs_stats = add_age_excl_col(data, n_indvs_stats)
    data, n_indvs_stats = add_hbb_col(data, n_indvs_stats)
    data = add_edge_info(data)
    diags = get_diag_med_data(args.diag_regex, args.med_regex)
    if args.diag_regex != "": data = add_first_diag_info(data, diags, args.diag_regex)
    if args.med_regex != "": data = add_first_med_info(data, diags, args.med_regex)

    # Saving
    if args.med_regex == "": 
        data["MED"] = np.nan
        data["MED_DATE"] = np.nan
    meta_data = data[["FINNGENID", "SEX", "FIRST_LAST", "N_YEAR", "N_MEASURE", "AGE_EXCLUDE", "HBB", "DIAG", "DIAG_DATE", "MED", "MED_DATE"]].drop_duplicates()
    data = data[["FINNGENID", "EVENT_AGE", "DATE", "VALUE", "ABNORM", "EDGE"]]
    
    data.to_csv(args.res_dir + file_name + ".csv", sep=",", index=False)
    meta_data.to_csv(args.res_dir + file_name + "_meta.csv", sep=",", index=False)
    n_indvs_stats.to_csv(log_dir + file_name + "_counts.csv", sep=",", index=False)
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
