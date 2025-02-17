# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/"))
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
from sklearn.model_selection import train_test_split

"""Adds SET column to data based on random split of individuals."""
def add_set(data, test_pct=0.1, valid_pct=0.1):
    data_train, data_rest = train_test_split(data, shuffle=True, random_state=3291, test_size=(valid_pct+test_pct), train_size=1-(valid_pct+test_pct), stratify=data.y_DIAG)
    print(f"N rows {len(data_train)}   N indvs {len(set(data_train.FINNGENID))}  N cases {sum(data_train.y_DIAG)} pct cases {round(sum(data_train.y_DIAG)/len(data_train), 2)}")
    print(f"N rows {len(data_rest)}   N indvs {len(set(data_rest.FINNGENID))}  N cases {sum(data_rest.y_DIAG)} pct cases {round(sum(data_rest.y_DIAG)/len(data_rest), 2)}")

    data_valid, data_test = train_test_split(data_rest, shuffle=True, random_state=391, test_size=test_pct/(test_pct+valid_pct), train_size=valid_pct/(test_pct+valid_pct), stratify=data_rest.y_DIAG)
    print(f"N rows {len(data_valid)}   N indvs {len(set(data_valid.FINNGENID))}  N cases {sum(data_valid.y_DIAG)} pct cases {round(sum(data_valid.y_DIAG)/len(data_valid), 2)}")
    print(f"N rows {len(data_test)}   N indvs {len(set(data_test.FINNGENID))}  N cases {sum(data_test.y_DIAG)} pct cases {round(sum(data_test.y_DIAG)/len(data_test), 2)}")

    data.loc[data.FINNGENID.isin(data_train.FINNGENID),"SET"] = 0
    data.loc[data.FINNGENID.isin(data_valid.FINNGENID),"SET"] = 1
    data.loc[data.FINNGENID.isin(data_test.FINNGENID),"SET"] = 2
    print(data.SET.value_counts(dropna=False))
    return(data)

def add_measure_counts(data):
    data.DATE = data.DATE.astype("datetime64[ns]")
    n_measures = data.groupby("FINNGENID").agg({"DATE": [("N_YEAR", lambda x: len(set(x.dt.year)))], "VALUE": len}).reset_index()
    n_measures.columns = ["FINNGENID", "N_YEAR", "N_MEASURE"]
    data = pd.merge(data, n_measures, on="FINNGENID", how="left")
    return(data)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/data/processed_data/step4_labels/")
    parser.add_argument("--data_path", type=str, help="Path to data.", default="/home/ivm/valid/data/processed_data/step3_abnorm/")
    parser.add_argument("--file_name", type=str, help="Path to data.")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--test_pct", type=float, help="Percentage of test data.", default=0.2)
    parser.add_argument("--valid_pct", type=float, help="Percentage of validation data.", default=0.2)

    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.file_name, logger, args)

    ### Getting Data
    data = pd.read_csv(args.data_path + args.file_name + ".csv", sep=",")
    metadata = pd.read_csv(args.data_path + args.file_name + "_meta.csv", sep=",")
    data = pd.merge(data, metadata, on="FINNGENID", how="left")

    ### Processing
    data = add_measure_counts(data)
    fids = set(data.loc[data.N_YEAR==1].FINNGENID)
    logging.info("Removed " + str(len(fids)) + " individuals with less than two years data")
    data = data.loc[~data.FINNGENID.isin(fids)].copy()

    # Making sure people are diagnosed and have data-based diagnosis AND removing all data before the first abnormal that lead to the data-based diag
    cases = data.loc[np.logical_and(data.DATE < data.FIRST_DIAG_DATE, data.DATE < data.DATA_FIRST_DIAG_ABNORM_DATE)]
    cases = cases.assign(y_DIAG = 1)
    # Start date is either the first diag date or the first abnormal that lead to the diagnosis
    cases = cases.assign(START_DATE=cases[["FIRST_DIAG_DATE", "DATA_FIRST_DIAG_ABNORM_DATE"]].values.min(axis=1))
    case_ages = data.loc[data.FINNGENID.isin(cases.FINNGENID)].query("DATE==DATA_FIRST_DIAG_ABNORM_DATE").rename(columns={"EVENT_AGE": "END_AGE", "VALUE": "y_MEAN"})[["FINNGENID", "END_AGE", "y_MEAN"]].drop_duplicates().groupby(["FINNGENID"]).head(1)
    print(case_ages.groupby("FINNGENID").filter(lambda x: len(x)>1))
    print(f"Row cases {len(case_ages)} and individuals {len(set(case_ages.FINNGENID))}")
    cases = pd.merge(cases, case_ages, on="FINNGENID", how="left")
    
    controls = data.loc[data.FIRST_DIAG_DATE.isnull()]
    controls_end_data = controls.sort_values(["FINNGENID", "DATE"], ascending=False).groupby("FINNGENID").head(1)[["FINNGENID", "DATE", "EVENT_AGE", "VALUE"]].rename(columns={"DATE":"START_DATE", "EVENT_AGE": "END_AGE", "VALUE": "y_MEAN"})
    controls = pd.merge(controls, controls_end_data, on="FINNGENID", how="left")
    controls = controls.loc[controls.DATE<controls.START_DATE]
    controls = controls.assign(y_DIAG = 0)

    new_data = pd.concat([cases, controls]) 
    labels = new_data[["FINNGENID", "y_DIAG", "END_AGE", "y_MEAN", "START_DATE"]].rename(columns={"END_AGE": "EVENT_AGE"}).drop_duplicates()
    labels = add_set(labels, args.test_pct, args.valid_pct)
    print(labels)
    print(f"N rows {len(new_data)}   N indvs {len(labels)}  N cases {sum(labels.y_DIAG)} pct cases {round(sum(labels.y_DIAG)/len(labels), 2)}")

    ### Saving
    new_data[["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"]].to_csv(args.res_dir + args.file_name + "_data-diag_data.csv", sep=",", index=False)
    labels.to_csv(args.res_dir + args.file_name + "_data-diag_labels.csv", sep=",", index=False)
    pd.DataFrame({"FINNGENID":list(fids)}).to_csv(args.res_dir + "logs/" + args.file_name + "_data-diag_removed_fids.csv", sep=",")
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())