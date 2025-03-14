# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from processing_utils import get_abnorm_func_based_on_name, add_measure_counts, add_set
from utils import *
# Standard stuff
import numpy as np
import pandas as pd
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
from datetime import datetime, date

def get_controls(data, months_buffer=0):
    """Get controls based on the data.
         Controls are defined as individuals without a diagnosis and data-based diagnosis."""
    # Removing all individuals with a dia   
    controls = data.loc[data.FIRST_DIAG_DATE.isnull()].copy()
    # Figuring out their last measurements (mostly for plotting in the end - predicting control status)
    controls_end_data = controls.sort_values(["FINNGENID", "DATE"], ascending=False).groupby("FINNGENID").head(1)[["FINNGENID", "DATE", "EVENT_AGE", "VALUE", "ABNORM_CUSTOM"]].rename(columns={"DATE":"START_DATE", "EVENT_AGE": "END_AGE", "VALUE": "y_MEAN", "ABNORM_CUSTOM": "LAST_ABNROM"})
    controls = pd.merge(controls, controls_end_data, on="FINNGENID", how="left")
    
    # only keeping those without any abnormal measurements
    remove_fids = controls.groupby("FINNGENID").filter(lambda x: sum(x.ABNORM_CUSTOM)>=1).FINNGENID
    controls = controls.loc[~controls.FINNGENID.isin(remove_fids)]
    
    # Removing all data in buffer of `months_buffer` months before first measurement
    controls = controls.assign(START_DATE=controls.START_DATE.astype("datetime64[ns]") - pd.DateOffset(months=months_buffer))

    # Setting control status
    controls = controls.assign(y_DIAG = 0)
    return(controls, remove_fids)

def get_cases(data, months_buffer=0):
    """Get cases based on the data.
       Cases are defined as individuals with a diagnosis and data-based diagnosis. 
       The start date is the first abnormality that lead to the diagnosis or the first diagnosis date."""
    # Making sure people are diagnosed and have data-based diagnosis AND removing all data before the first abnormal that lead to the data-based diag
    cases = data.loc[np.logical_and(data.DATE < data.FIRST_DIAG_DATE, data.DATE < data.DATA_FIRST_DIAG_ABNORM_DATE)]
    # only keeping those without any abnormal measurements
    #cases = cases.loc[~cases.FINNGENID.isin(remove_fids)]
    cases = cases.assign(y_DIAG = 1)
    # Start date is either the first diag date or the first abnormal that lead to the diagnosis minus a buffer of `months_buffer` months
    cases = cases.assign(START_DATE=cases[["FIRST_DIAG_DATE", "DATA_FIRST_DIAG_ABNORM_DATE"]].values.min(axis=1))
    cases = cases.assign(START_DATE=cases.START_DATE.astype("datetime64[ns]") - pd.DateOffset(months=months_buffer))
    # Age and value at first abnormality that lead to the data-based diagnosis
    case_ages = data.loc[data.FINNGENID.isin(cases.FINNGENID)].query("DATE==DATA_FIRST_DIAG_ABNORM_DATE").rename(columns={"EVENT_AGE": "END_AGE", "VALUE": "y_MEAN"})[["FINNGENID", "END_AGE", "y_MEAN"]].drop_duplicates().groupby(["FINNGENID"]).head(1)
    cases = pd.merge(cases, case_ages, on="FINNGENID", how="left")
    
    return(cases)


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/data/processed_data/step4_labels/")
    parser.add_argument("--data_path", type=str, help="Path to data.", default="/home/ivm/valid/data/processed_data/step3_abnorm/")
    parser.add_argument("--file_name", type=str, help="Path to data.")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--test_pct", type=float, help="Percentage of test data.", default=0.2)
    parser.add_argument("--valid_pct", type=float, help="Percentage of validation data.", default=0.2)
    parser.add_argument("--min_n_years", type=float, help="Minimum number of years with at least one measurement to be included", default=2)
    parser.add_argument("--months_buffer", type=int, help="Minimum number months before prediction to be removed.", default=2)

    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.file_name + "_" + get_datetime(), logger, args)

    ### Getting Data
    data = pd.read_csv(args.data_path + args.file_name + ".csv", sep=",")
    metadata = pd.read_csv(args.data_path + args.file_name + "_meta.csv", sep=",")
    data = pd.merge(data, metadata, on="FINNGENID", how="left")

    ### Processing
    if args.min_n_years > 1:
        data = add_measure_counts(data)
        fids = set(data.loc[data.N_YEAR==1].FINNGENID)
        logging.info("Removed " + str(len(fids)) + " individuals with less than two years data")
        data = data.loc[~data.FINNGENID.isin(fids)].copy()
        pd.DataFrame({"FINNGENID":list(fids)}).to_csv(args.res_dir + "logs/" + args.file_name + "_data-diag-noabnorm_" + get_date() + "_removed_fids.csv", sep=",")

    cases = get_cases(data, args.months_buffer)
    controls, ctrl_remove_fids = get_controls(data, args.months_buffer)
    new_data = pd.concat([cases, controls]) 

    # Data only before the start date
    new_data = new_data.loc[new_data.DATE < new_data.START_DATE]    
    remove_fids = new_data.groupby("FINNGENID").filter(lambda x: sum(x.ABNORM_CUSTOM)>=1).FINNGENID
    remove_fids = set(remove_fids) | set(ctrl_remove_fids)
    new_data = new_data.loc[~new_data.FINNGENID.isin(remove_fids)].copy()
    logging.info("Removed " + str(len(remove_fids)) + " individuals due to potential loss of followup")
    pd.DataFrame({"FINNGENID":list(set(remove_fids))}).to_csv(args.res_dir + "logs/" + args.file_name + "_data-diag-noabnorm_" + get_date() + "_removed_lof_fids.csv", sep=",")

    # Labels for all
    labels = new_data[["FINNGENID", "y_DIAG", "END_AGE", "SEX","y_MEAN", "START_DATE"]].rename(columns={"END_AGE": "EVENT_AGE"}).drop_duplicates()
    labels = add_set(labels, args.test_pct, args.valid_pct)
    logging_print(f"N rows {len(new_data)}   N indvs {len(labels)}  N cases {sum(labels.y_DIAG)} pct cases {round(sum(labels.y_DIAG)/len(labels), 2)*100}%")

    new_data[["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"]].to_csv(args.res_dir + args.file_name + "_data-diag-noabnorm_" + get_date() + ".csv", sep=",", index=False)
    labels.to_csv(args.res_dir + args.file_name + "_data-diag-noabnorm_" + get_date() + "_labels.csv", sep=",", index=False)
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())