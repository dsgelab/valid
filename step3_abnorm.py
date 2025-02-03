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

"""First Abnormality"""
def add_first_abnorm_info(data):
    earliest_abnorm_fg = data.loc[data.ABNORM != 0].groupby("FINNGENID").agg({"DATE": "min"}).reset_index().rename(columns={"DATE": "FIRST_ABNORM_DATE"})

    data = pd.merge(data, earliest_abnorm, on="FINNGENID", how="left")
    data = pd.merge(data, earliest_abnorm_fg, on="FINNGENID", how="left")

    return(data)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/data/processed_data/step3/")
    parser.add_argument("--file_path", type=str, help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end", default="/home/ivm/valid/data/processed_data/step2/")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--source_file_date", type=str, help="Date of file.", required=True)

    args = parser.parse_args()

    return(args)
 

if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    
    log_dir = args.res_dir + "logs/"
    date = datetime.today().strftime("%Y-%m-%d")
    date_time = datetime.today().strftime("%Y-%m-%d-%H%M")
    file_name = args.lab_name + "_" + date
    file_path_data = args.file_path + args.lab_name + "_" + args.source_file_date + ".csv"
    file_path_meta = args.file_path + args.lab_name + "_" + args.source_file_date + "_meta.csv"
    log_file_name = args.lab_name + "_" + date_time
    make_dir(log_dir)
    make_dir(args.res_dir)
    
    init_logging(log_dir, log_file_name, date_time, logger)

    ### Getting Data
    data = pd.read_csv(file_path_data, sep=",")
    metadata = pd.read_csv(file_path_meta, sep=",")
    data = pd.merge(data, metadata, on="FINNGENID", how="left")

    ### Processing
    # Age limits
    data = data.query("AGE_EXCLUDE == 0").drop("AGE_EXCLUDE", axis=1)
    # abnormality
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
        
    data = get_abnorm_func_based_on_name(args.lab_name)(data)
        
    else: data = three_level_abnorm(data)
    # Metadata
    data = add_first_abnorm_info(data)
    # Saving
    meta_data = data[["FINNGENID", "SEX", "FIRST_LAST", "N_YEAR", "N_MEASURE", "HBB", "DIAG", "DIAG_DATE", "MED", "MED_DATE", "FIRST_ABNORM_DATE", "FIRST_ABNORM_CUSTOM_DATE"]].drop_duplicates()
    data = data[["FINNGENID", "EVENT_AGE", "DATE", "VALUE", "ABNORM", "ABNORM_CUSTOM", "EDGE"]]
    
    data.to_csv(args.res_dir + file_name + ".csv", sep=",", index=False)
    meta_data.to_csv(args.res_dir + file_name + "_meta.csv", sep=",", index=False)
    
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())