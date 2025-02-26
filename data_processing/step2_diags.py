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


"""Get medication and diagnosis information based on the diagnosis and medication regex given."""
def get_diag_med_data(diag_regex,
                      med_regex="",
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
    
"""First exclusion"""
def get_codes_first(data, crnt_regex):
    data = data.loc[data.CODE.str.contains(crnt_regex)]
    data = data.sort_values(["FINNGENID", "APPROX_EVENT_DAY"], ascending=True).groupby(["FINNGENID", "CODE"]).head(1)
    return(data)
   
def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory (step2)", default="/home/ivm/valid/data/processed_data/step2_clean/")
    parser.add_argument("--lab_name", help="Readable name of the measurement value.", required=True)
    parser.add_argument("--diag_regex", type=str, help="Regex for selecting Code1 and 2 from the service sector detailed longitudinal data.", required=True)
    parser.add_argument("--med_regex", type=str, help="Regex for selecting medication purchases.", required=False, default="")
    parser.add_argument("--diag_excl_regex", type=str, help="Regex for selecting Code1 and 2 from the service sector detailed longitudinal data.", required=True)

    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    #### Preparations
    timer = Timer()
    args = get_parser_arguments()

    make_dir(args.res_dir);
    init_logging(args.res_dir, args.lab_name, logger, args)

    #### Data processing
    # Raw dat
    if args.diag_excl_regex != "" or args.med_excl_regex != "": 
        diags = get_diag_med_data(args.diag_excl_regex)
        excls = get_codes_first(diags, args.diag_excl_regex).rename({"APPROX_EVENT_DAY": "EXCL_DATE", "CODE":"EXCL_CODE"})
        excls.to_csv(args.res_dir + args.lab_name + "_excls.csv", sep=",", index=False)

    if args.diag_regex != "" or args.med_regex != "":
        all_diags = get_diag_med_data(args.diag_regex, args.med_regex)
        if args.diag_regex != "": 
            icd_diags = get_codes_first(all_diags, args.diag_regex).rename({"APPROX_EVENT_DAY": "DIAG_DATE", "CODE":"DIAG"})
            icd_diags.to_csv(args.res_dir + args.lab_name + "_diags.csv", sep=",", index=False)
        if args.med_regex != "": 
            med_diags = get_codes_first(all_diags, args.med_regex).rename({"APPROX_EVENT_DAY": "MED_DATE", "CODE":"MED"})
            med_diags.to_csv(args.res_dir + args.lab_name + "_meds.csv", sep=",", index=False)

    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
    