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
from processing_utils import *

def egfr_transform(data):
    data.loc[np.logical_and(data["VALUE"] <= 62, data["SEX"] == "female"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] <= 62, data["SEX"] == "female"), "VALUE"]/61.9 )**(-0.329))*(0.993**data.loc[np.logical_and(data["VALUE"] <= 62, data["SEX"] == "female"), "EVENT_AGE"])*144
    data.loc[np.logical_and(data["VALUE"] > 62, data["SEX"] == "female"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] > 62, data["SEX"] == "female"), "VALUE"]/61.9 )**(-1.209))*(0.993**data.loc[np.logical_and(data["VALUE"] > 62, data["SEX"] == "female"), "EVENT_AGE"])*144
    data.loc[np.logical_and(data["VALUE"] <= 80, data["SEX"] == "male"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] <= 80, data["SEX"] == "male"), "VALUE"]/79.6 )**(-0.411))*(0.993**data.loc[np.logical_and(data["VALUE"] <= 80, data["SEX"] == "male"), "EVENT_AGE"])*141
    data.loc[np.logical_and(data["VALUE"] > 80, data["SEX"] == "male"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] > 80, data["SEX"] == "male"), "VALUE"]/79.6 )**(-1.209))*(0.993**data.loc[np.logical_and(data["VALUE"] > 80, data["SEX"] == "male"), "EVENT_AGE"])*141
    data.VALUE = data.VALUE_TRANSFORM
    return(data)
    
"""First Abnormality"""
def add_first_abnorm_info(data):
    earliest_abnorm = data.loc[data.ABNORM_CUSTOM != 0].groupby("FINNGENID").agg({"DATE": "min"}).reset_index().rename(columns={"DATE": "FIRST_ABNORM_CUSTOM_DATE"})
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

def init_logging(log_dir, log_file_name, date_time):
    logging.basicConfig(filename=log_dir+log_file_name+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.info("Time: " + date_time + " Args: --" + ' --'.join(f'{k}={v}' for k, v in vars(args).items()))
     
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
    
    init_logging(log_dir, log_file_name, date_time)

    ### Getting Data
    data = pd.read_csv(file_path_data, sep=",")
    metadata = pd.read_csv(file_path_meta, sep=",")
    data = pd.merge(data, metadata, on="FINNGENID", how="left")

    ### Processing
    # Age limits
    data = data.query("AGE_EXCLUDE == 0").drop("AGE_EXCLUDE", axis=1)
    # ABNORMity
    if args.lab_name == "tsh":
        data = three_level_abnorm(data)
        data = tsh_abnorm(data)
    if args.lab_name == "hba1c": 
        data = simple_abnorm(data)
        data = hba1c_abnorm(data)
    if args.lab_name == "ldl":
        data = simple_abnorm(data)
        data = ldl_abnorm(data)
    if args.lab_name == "egfr" or args.lab_name == "krea": 
        data = egfr_transform(data)
        data = simple_abnorm(data)
        data = egfr_abnorm(data)
    if args.lab_name == "cyst":
        data = cystc_abnorm(data)
        data = simple_abnorm(data)
    if args.lab_name == "gluc" or args.lab_name=="fgluc":
        data = gluc_abnorm(data)
        data = three_level_abnorm(data)
        
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

#python3 /home/ivm/valid/scripts/step3_abnorm.py --source_file_date=2024-10-16 --lab_name=krea
#python3 /home/ivm/valid/scripts/step3_abnorm.py --source_file_date=2024-10-17 --lab_name=krea
