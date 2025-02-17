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

"""Adds field START_DATE to data based on x years before last measurement for each individual."""
def add_start_of_pred_period(data, pred_len):
    start_pred_period = data[["FINNGENID", "EDGE", "DATE"]]
    # Last measurement
    start_pred_period = start_pred_period.loc[start_pred_period.EDGE == 1]
    # Remove len of period in years
    start_pred_period.DATE = start_pred_period.DATE.astype("datetime64[ns]")
    start_pred_period.DATE = start_pred_period.DATE - pd.DateOffset(years=pred_len)
    start_pred_period = start_pred_period.rename(columns={"DATE": "START_DATE"}).drop_duplicates()
    # Add to data
    data = pd.merge(data, start_pred_period[["FINNGENID", "START_DATE"]], on="FINNGENID", how="left")
    return(data)
    
def get_carry_over_preds(data):
    data = data.loc[data.DATE < data.START_DATE]
    data = data.sort_values(["FINNGENID", "DATE"]).groupby("FINNGENID").tail(1)
    data = data.loc[data.ABNORM.notnull()]
    return(data)
    
def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/data/processed_data/step4/")
    parser.add_argument("--file_path", type=str, help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end", default="/home/ivm/valid/data/processed_data/step3/")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--source_file_date", type=str, help="Date of file.", required=True)
    parser.add_argument("--pred_len", type=int, help="Length of period in years before last measure for pred window.", default=1)
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
    file_name = args.lab_name + "_preds_" + date
    file_path_data = args.file_path + args.lab_name + "_" + args.source_file_date + ".csv"
    file_path_meta = args.file_path + args.lab_name + "_" + args.source_file_date + "_meta.csv"
    log_file_name = args.lab_name + "_preds_" + date_time
    make_dir(log_dir)
    make_dir(args.res_dir)
    
    init_logging(log_dir, log_file_name, date_time)

    ### Getting Data
    data = pd.read_csv(file_path_data, sep=",")
    metadata = pd.read_csv(file_path_meta, sep=",")
    data = pd.merge(data, metadata, on="FINNGENID", how="left")
    
    ### Processing
    data = add_start_of_pred_period(data, args.pred_len)

    preds = get_carry_over_preds(data)
    preds = preds[["FINNGENID", "EVENT_AGE", "DATE", "VALUE", "ABNORM", "ABNORM_CUSTOM"]].drop_duplicates()
    preds.to_csv(args.res_dir + file_name + "_" + str(args.pred_len) + "-year_carryover.csv", sep=",", index=False)
    