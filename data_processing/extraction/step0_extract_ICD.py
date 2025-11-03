# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import *
# Standard stuff
import pandas as pd
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
from datetime import datetime

def get_death_data(fg_ver="r12"):
    table = f"`finngen-production-library.sandbox_tools_{fg_ver}.finngen_{fg_ver}_service_sector_detailed_longitudinal_v1`"

    query_death = f""" SELECT FINNGENID, EVENT_AGE, APPROX_EVENT_DAY, CODE1, CATEGORY, ICDVER, SOURCE
                       FROM {table}
                       WHERE SOURCE="DEATH" AND CODE1 != "None" AND REGEXP_CONTAINS(CODE1, "^[A-Z][0-9]+")
                 """
    death_diags = query_to_df(query_death)
    death_diags = death_diags.rename({"APPROX_EVENT_DAY": "DATE", "CODE1": "ICD_CODE", "ICDVER": "ICD_VERSION"})
    logger.info("Time import death: "+timer.get_elapsed())
    print(death_diags)
    return(death_diags)


def get_pat_data(fg_ver="r12"):
    table = f"`finngen-production-library.sandbox_tools_{fg_ver}.finngen_{fg_ver}_service_sector_detailed_longitudinal_v1`"
    query_pat = f""" SELECT FINNGENID, EVENT_AGE, APPROX_EVENT_DAY, CODE1, CODE2, CATEGORY, ICDVER, SOURCE
                     FROM {table}
                      WHERE (SOURCE IN ("PRIM_OUT", "INPAT", "OUTPAT")) AND 
                            (CODE2 != "None" OR CODE1 != "None") AND 
                            (REGEXP_CONTAINS(CODE1, "^[A-Z][0-9]+") OR REGEXP_CONTAINS(CODE2, "^[A-Z][0-9]+")) AND
                            (NOT (REGEXP_CONTAINS(CATEGORY, "^ICP") OR REGEXP_CONTAINS(CATEGORY, "^OP") OR REGEXP_CONTAINS(CATEGORY, "^MOP") OR CATEGORY = "None"))
                 """
    pat_diags = query_to_df(query_pat)
    logger.info("Time outpat, inpat, prim_out import: "+timer.get_elapsed())
    
    # CODE1 is diagnosis and CODE2 symptom
    pat_diags = pat_diags.unpivot(index=["FINNGENID", "EVENT_AGE", "APPROX_EVENT_DAY", "CATEGORY", "ICDVER", "SOURCE"])

    pat_diags = pat_diags.drop("variable").unique()
    pat_diags = pat_diags.rename({"APPROX_EVENT_DAY": "DATE", "ICDVER":"ICD_VERSION","value":"ICD_CODE"})

    pat_diags = pat_diags.filter(~pl.col.ICD_CODE.is_null(), pl.col.ICD_CODE.str.contains("^[A-Z][0-9]"))
    print(pat_diags)
    return(pat_diags)

def get_med_data(fg_ver="r12"):
    table = f"`finngen-production-library.sandbox_tools_{fg_ver}.finngen_{fg_ver}_service_sector_detailed_longitudinal_v1`"

    timer = Timer()
    query_meds = f""" SELECT FINNGENID, EVENT_AGE, APPROX_EVENT_DAY, CODE2, CATEGORY, ICDVER, SOURCE
                      FROM {table}
                      WHERE (SOURCE= "REIMB") AND (CODE2 != "None") AND (REGEXP_CONTAINS(CODE2, "^[A-Z][0-9]+"))
                 """
    med_diags = query_to_df(query_meds)
    med_diags = med_diags.rename({"APPROX_EVENT_DAY": "DATE", "CODE2": "ICD_CODE", "ICDVER": "ICD_VERSION"})
    med_diags = med_diags.with_columns(CATEGORY=pl.lit("1"))
    print(med_diags)
    logger.info("Time med diags import: "+timer.get_elapsed())
    return(med_diags)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_ver", help="FinnGen release version", default="r12")
    parser.add_argument("--res_dir", type=str, help="Where data should be save", default="/home/ivm/PheRS/data/processed_data/step0/")

    args = parser.parse_args()

    return(args)

def init_logging(log_dir, log_file_name, date_time):
    logging.basicConfig(filename=log_dir+log_file_name+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.info("Time: " + date_time + " Args: --" + ' --'.join(f'{k}={v}' for k, v in vars(args).items()))
     
if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    
    log_dir = args.res_dir + "logs/"
    date = datetime.today().strftime("%Y-%m-%d")
    date_time = datetime.today().strftime("%Y-%m-%d-%H%M")
    file_name = "ICD_long_" + args.fg_ver + "_" + date
    log_file_name = "ICD_long_" + args.fg_ver + "_" +date_time
    make_dir(args.res_dir)
    make_dir(log_dir)
    
    init_logging(log_dir, log_file_name, date_time)
    death_diags = get_death_data(args.fg_ver)
    pat_diags = get_pat_data(args.fg_ver)
    med_diags = get_med_data(args.fg_ver)
    all_diags = pl.concat([death_diags.select(pat_diags.columns), pat_diags, med_diags.select(pat_diags.columns)])
    
    all_diags.write_parquet(args.res_dir + file_name + ".parquet")
    logger.info(all_diags["ICD_VERSION"].value_counts())
    logger.info(all_diags["SOURCE"].value_counts())

#python3 /home/ivm/PheRS/scripts/extract/step0_extract_ICD.py --fg_ver r13
