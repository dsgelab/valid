# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from processing_utils import get_abnorm_func_based_on_name
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

"""Gets abnormality and values for last x years for individuals as goal value/labels for prediction.
   Two different sets of abnormality predictions for FG based abnormality and custom."""
def get_labels(data, abnorm_func):
    # Mean, median, max, min, and abnormal in that period
    # abnormal function needs to be send. I.e. for eGFR it will be "min" and for HbA1c "max"
    labels = data.loc[data.DATE >= data.START_DATE].sort_values(["FINNGENID", "DATE"], ascending=True).groupby("FINNGENID").agg({"EVENT_AGE": "max", "VALUE": ["mean", "median", "max", "min"], "START_DATE": "first"}).reset_index()
    labels.columns = ["FINNGENID", "EVENT_AGE", "y_MEAN", "y_MEDIAN", "y_MAX", "y_MIN", "START_DATE"]
    ## Adding abnormality
    labels = abnorm_func(labels, "y_MEAN").rename(columns={"ABNORM_CUSTOM": "y_MEAN_ABNORM"})
    labels = abnorm_func(labels, "y_MEDIAN").rename(columns={"ABNORM_CUSTOM": "y_MEDIAN_ABNORM"})
    labels = abnorm_func(labels, "y_MAX").rename(columns={"ABNORM_CUSTOM": "y_MAX_ABNORM"})
    labels = abnorm_func(labels, "y_MIN").rename(columns={"ABNORM_CUSTOM": "y_MIN_ABNORM"})

    return(labels)

"""Adds SET column to data based on random split of individuals."""
def add_set(data, test_pct=0.2, valid_pct=0.2):
    fg_map = dict([(x,y) for x,y in enumerate(sorted(set(data["FINNGENID"])))])
    n_indvs = len(fg_map)
    
    np.random.seed(1053)
    indv_idxs = np.random.permutation(n_indvs)
    n_test = int(n_indvs * test_pct)
    n_valid = int(n_indvs * valid_pct)
    
    test_ids = [fgid for idx, fgid in fg_map.items() if idx in indv_idxs[:n_test]]
    train_ids = [fgid for idx, fgid in fg_map.items() if idx in indv_idxs[n_test+n_valid:]]
    valid_ids = [fgid for idx, fgid in fg_map.items() if idx in indv_idxs[n_test:n_test+n_valid]]
    
    data.loc[data.FINNGENID.isin(train_ids), "SET"] = 0
    data.loc[data.FINNGENID.isin(test_ids), "SET"] = 2
    data.loc[data.FINNGENID.isin(valid_ids), "SET"] = 1
    return(data)


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/data/processed_data/step4_labels/")
    parser.add_argument("--file_path", type=str, help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end", default="/home/ivm/valid/data/processed_data/step3_abnorm/")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--source_file_date", type=str, help="Date of file.", required=True)
    parser.add_argument("--pred_len", type=int, help="Length of period in years before last measure for pred window.", default=1)
    parser.add_argument("--test_pct", type=int, help="Percentage of test data.", default=0.2)
    parser.add_argument("--valid_pct", type=int, help="Percentage of validation data.", default=0.2)


    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    
    log_dir = args.res_dir + "logs/"
    date = datetime.today().strftime("%Y-%m-%d")
    date_time = datetime.today().strftime("%Y-%m-%d-%H%M")
    file_name = args.lab_name + "_" + args.source_file_date + "_labels_" + date
    file_path_data = args.file_path + args.lab_name + "_" + args.source_file_date + ".csv"
    file_path_meta = args.file_path + args.lab_name + "_" + args.source_file_date + "_meta.csv"
    log_file_name = args.lab_name + "_labels_" + date_time
    make_dir(log_dir)
    make_dir(args.res_dir)
    
    init_logging(log_dir, log_file_name, logger, args)

    ### Getting Data
    data = pd.read_csv(file_path_data, sep=",")
    metadata = pd.read_csv(file_path_meta, sep=",")
    data = pd.merge(data, metadata, on="FINNGENID", how="left")
    
    ### Processing
    data = add_start_of_pred_period(data, args.pred_len)
    labels = get_labels(data, get_abnorm_func_based_on_name(args.lab_name))
    labels = add_set(labels, args.test_pct, args.valid_pct)

    ### Saving
    labels.to_csv(args.res_dir + file_name + "_" + str(args.pred_len) + "-year.csv", sep=",", index=False)

    #Final logging
    logger.info("Time total: "+timer.get_elapsed())