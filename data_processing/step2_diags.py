 # Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer, query_to_df, get_date, get_datetime, make_dir, init_logging, logging_print, read_file
# Standard stuff
import numpy as np
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
from datetime import datetime


def get_diag_med_data(diag_regex,
                      med_regex="",
                      fg_ver = "R12"):
    """Get medication and diagnosis information based on the diagnosis and medication regex given."""

    if fg_ver == "R12":
        table = f"`finngen-production-library.sandbox_tools_r12.finngen_r12_service_sector_detailed_longitudinal_v1`"
    elif fg_ver == "R13":
        table = f"`finngen-production-library.sandbox_tools_r13.finngen_r13_service_sector_detailed_longitudinal_v1`"
    else:
        raise ValueError("Finngen version must be R12 or R13.")
    timer = Timer()

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
    diags = diags.unpivot(index=["FINNGENID", "EVENT_AGE", "APPROX_EVENT_DAY"])
    diags = diags.drop(["variable"]).unique()
    diags = diags.rename({"value":"CODE"})
    # Other codes are numbers
    diags = diags.filter(pl.col("CODE").is_not_null())
    diags = diags.filter(pl.col("CODE").str.contains("^[A-Z][0-9]"))
    logging_print("Time import: "+timer.get_elapsed())
    return(diags)
    
"""First exclusion"""
def get_codes_first(data: pl.DataFrame,
                    crnt_regex: str) -> pl.DataFrame:
    """Get the first occurance of a code for each individual and code."""
    data = data.filter(pl.col("CODE").str.contains(crnt_regex))
    data = (data
            .sort(["FINNGENID", "APPROX_EVENT_DAY"], descending=False)
            .group_by(["FINNGENID", "CODE"])
            .head(1))
    return(data)
   
def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Info for reading and naming
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--fg_ver", type=str, help="Finngen version (R12 or R13)", required=True)

    # Regex for selecting the data
    parser.add_argument("--diag_regex", type=str, help="Regex for selecting Code1 and 2 from the service sector detailed longitudinal data.", required=True)
    parser.add_argument("--med_regex", type=str, help="Regex for selecting medication purchases.", required=False, default="")
    parser.add_argument("--diag_excl_regex", type=str, help="Regex for selecting Code1 and 2 from the service sector detailed longitudinal data.", required=True)


    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    out_file_path = args.res_dir + args.lab_name + "_" + args.fg_ver + "_" + get_date()
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.lab_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Exclusion data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.diag_excl_regex != "" or args.med_excl_regex != "": 
        diags = get_diag_med_data(diag_regex=args.diag_excl_regex, 
                                  fg_ver=args.fg_ver)
        excls = (get_codes_first(diags, args.diag_excl_regex)
                 .rename({"APPROX_EVENT_DAY": "EXCL_DATE", "CODE":"EXCL_CODE"}))
        print(excls)
        excls.write_parquet(out_file_path + "_excls.parquet")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Diagnosis data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.diag_regex != "" or args.med_regex != "":
        all_diags = get_diag_med_data(diag_regex=args.diag_regex, 
                                      med_regex=args.med_regex, 
                                      fg_ver=args.fg_ver)
        if args.diag_regex != "": 
            icd_diags = (get_codes_first(all_diags, args.diag_regex)
                         .rename({"APPROX_EVENT_DAY": "DIAG_DATE", "CODE":"DIAG"}))
            print(icd_diags)
            icd_diags.write_parquet(out_file_path + "_diags.parquet")
        if args.med_regex != "": 
            med_diags = (get_codes_first(all_diags, args.med_regex)
                         .rename({"APPROX_EVENT_DAY": "MED_DATE", "CODE":"MED"}))
            print(med_diags)
            med_diags.write_parquet(out_file_path + "_meds.parquet")

    #Final logging
    logger.info("Time total: "+timer.get_elapsed())