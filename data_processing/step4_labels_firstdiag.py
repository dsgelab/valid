# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, get_datetime, make_dir, init_logging, Timer, read_file, logging_print
from processing_utils import add_measure_counts, add_set
# Standard stuff
import numpy as np
import pandas as pd
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse

def get_controls(data: pl.DataFrame,  
                 no_abnormal_ctrls=1,
                 months_buffer=0):
    """Get controls based on the data.
         Controls are defined as individuals without a diagnosis and data-based diagnosis."""
    # Removing all individuals with a dia   
    controls = data.filter(pl.col("DATA_DIAG_DATE").is_null())
    print
    # Figuring out their last measurements (mostly for plotting in the end - predicting control status)
    controls_end_data = (controls.sort(["FINNGENID", "DATE"], descending=True)
                                 .group_by("FINNGENID")
                                 .head(1)
                                 .select(["FINNGENID", "DATE", "EVENT_AGE", "VALUE", "ABNORM_CUSTOM"])
                                 .rename({"DATE":"START_DATE", "EVENT_AGE": "END_AGE", "VALUE": "y_MEAN", "ABNORM_CUSTOM": "LAST_ABNORM"}))
    controls_end_data = controls_end_data.with_columns(pl.col("START_DATE").dt.offset_by(f"-{months_buffer}mo").alias("START_DATE"))
    controls = controls.join(controls_end_data, on="FINNGENID", how="left")
    controls = controls.with_columns(y_DIAG=0)
    # Removing cases with prior abnormal data - if wanted
    if(no_abnormal_ctrls == 1):
        ctrl_prior_data = controls.filter(pl.col("DATE") < pl.col("START_DATE"))
        remove_fids = (ctrl_prior_data
                       .group_by("FINNGENID")
                       .agg(pl.col("ABNORM_CUSTOM").sum().alias("N_PRIOR_ABNORM"))
                       .filter(pl.col("N_PRIOR_ABNORM")>=1)
                       .get_column("FINNGENID"))
        controls = controls.filter(~pl.col("FINNGENID").is_in(remove_fids))
    else:
        # Always removing last abnormal without diagnosis because of fear of censoring
        remove_fids = controls_end_data.filter(pl.col("LAST_ABNORM")>=1).select("FINNGENID")
        remove_fids = set(remove_fids["FINNGENID"])
        controls = controls.filter(~pl.col("FINNGENID").is_in(remove_fids))

    return(controls, remove_fids)

def get_cases(data: pl.DataFrame, 
              no_abnormal_cases=1,
              months_buffer=0) -> pl.DataFrame:
    """Get cases based on the data.
       Cases are defined as individuals with a diagnosis and data-based diagnosis. 
       The start date is the first abnormality that lead to the diagnosis or the first diagnosis date."""
    # Cases have a diagnosis 
    cases = data.filter(pl.col("DATA_DIAG_DATE").is_not_null())

    # Start date is either the first diag date or the first abnormal that lead to the diagnosis minus a buffer of `months_buffer` months
    cases = cases.with_columns(y_DIAG=1, 
                               START_DATE=pl.min_horizontal(["FIRST_DIAG_DATE", "DATA_FIRST_DIAG_ABNORM_DATE"]))
    cases = cases.with_columns(pl.col("START_DATE").dt.offset_by(f"-{months_buffer}mo").alias("START_DATE"))

    # Removing cases with prior abnormal data - if wanted
    if(no_abnormal_cases == 1):
        case_prior_data = cases.filter(pl.col("DATE") < pl.col("START_DATE"))
        remove_fids = (case_prior_data
                       .group_by("FINNGENID")
                       .agg(pl.col("ABNORM_CUSTOM").sum().alias("N_PRIOR_ABNORM"))
                       .filter(pl.col("N_PRIOR_ABNORM")>=1)
                       .get_column("FINNGENID"))
        cases = cases.filter(~pl.col("FINNGENID").is_in(remove_fids))
    else:
        remove_fids = set()

    # Age and value at first abnormality that lead to the data-based diagnosis
    case_ages = (cases.filter(pl.col("DATE")==pl.col("START_DATE"))
                      .rename({"EVENT_AGE": "END_AGE", "VALUE": "y_MEAN", "ABNORM_CUSTOM": "LAST_ABNORM"})
                      .select(["FINNGENID", "END_AGE", "y_MEAN", "LAST_ABNORM"])
                      .unique())
    cases = cases.join(case_ages, on="FINNGENID", how="left")
    
    return(cases, remove_fids)


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument("--data_path", type=str, help="Path to data.", default="/home/ivm/valid/data/processed_data/step3_meta/")
    parser.add_argument("--file_name", type=str, help="Path to data.")
    parser.add_argument("--exclusion_path", type=str, help="Path to exclusions, atm excluding all FGIDs in there.")
    parser.add_argument("--exclude_diags", type=int, default=0, help="Whether to exclude individuals with prior diags or not. 0 = no, 1 = yes")

    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)

    # Settings for data-processing
    parser.add_argument("--test_pct", type=float, help="Percentage of test data.", default=0.2)
    parser.add_argument("--valid_pct", type=float, help="Percentage of validation data.", default=0.2)
    parser.add_argument("--no_abnorm", type=float, help="Whether to create super controls where all prior data is normal.", default=1)
    parser.add_argument("--min_n_years", type=float, help="Minimum number of years with at least one measurement to be included", default=2)
    parser.add_argument("--months_buffer", type=int, help="Minimum number months before prediction to be removed.", default=2)
    parser.add_argument("--min_age", type=float, help="Minimum age at prediction time", default=30)
    parser.add_argument("--max_age", type=float, help="Maximum age at prediction time", default=70)

    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    timer = Timer()
    args = get_parser_arguments()

    # Setting up logging
    if args.no_abnorm == 1:
        out_file_name = args.file_name + "_data-diag_noabnorm_" + get_date() 
    else:
        out_file_name = args.file_name + "_data-diag_" + get_date()

    make_dir(args.res_dir)
    init_logging(args.res_dir, args.file_name + "_" + get_datetime(), logger, args)
    make_dir(args.res_dir + "logs/removed_fids/")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = read_file(args.data_path + args.file_name + ".csv")
    metadata = read_file(args.data_path + args.file_name + "_meta.csv",
                         schema={"FIRST_DIAG_DATE": pl.Date, "DATA_FIRST_DIAG_ABNORM_DATE": pl.Date, "DATA_DIAG_DATE": pl.Date})
    data = data.join(metadata, on="FINNGENID", how="left")

    ### Processing
    if args.min_n_years > 1:
        data = add_measure_counts(data)
        fids = set(data.filter(pl.col("N_YEAR")==1)["FINNGENID"])
        logging.info("Removed " + str(len(fids)) + " individuals with less than two years data")
        data = data.filter(~pl.col("FINNGENID").is_in(fids))
        pd.DataFrame({"FINNGENID":list(fids)}).to_csv(args.res_dir + "logs/removed_fids/" + args.file_name + out_file_name + "_reason_nyears.csv", sep=",")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Cases and controls                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    cases, case_remove_fids = get_cases(data, args.no_abnorm, args.months_buffer)
    controls, ctrl_remove_fids = get_controls(data, args.no_abnorm, args.months_buffer)
    new_data = pl.concat([cases, controls.select(cases.columns)]) 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Age                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    age_remove_fids = set(new_data.filter((pl.col("END_AGE")<args.min_age)|(pl.col("END_AGE")>args.max_age))["FINNGENID"])
    new_data = new_data.filter(~pl.col("FINNGENID").is_in(age_remove_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Diag exclusions                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if args.exclude_diags == 1:
        exclusion_data = read_file(args.exclusion_path)
        diag_remove_fids = (exclusion_data
                            .join(new_data.select(["FINNGENID", "START_DATE"]), on="FINNGENID", how="left")
                            .filter(pl.col("APPROX_EVENT_DAY") < pl.col("START_DATE"))
                            .get_column("FINNGENID"))
        new_data = new_data.filter(~pl.col("FINNGENID").is_in(diag_remove_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 labels                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    labels = new_data.select(["FINNGENID", "y_DIAG", "END_AGE", "SEX","y_MEAN", "START_DATE"]).rename({"END_AGE": "EVENT_AGE"}).unique()
    labels = add_set(labels, args.test_pct, args.valid_pct)
    logging_print(f"N rows {len(new_data)}   N indvs {len(labels)}  N cases {sum(labels["y_DIAG"])} pct cases {round(sum(labels["y_DIAG"])/len(labels), 2)*100}%")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Data before start                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    new_data = new_data.filter(pl.col("DATE") < pl.col("START_DATE"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info("Removed " + str(len(age_remove_fids)) + " individuals with age < " + str(args.min_age) + " or > " + str(args.max_age))
    pd.DataFrame({"FINNGENID":list(set(age_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_age.csv", sep=",")
    logging.info("Removed " + str(len(set(case_remove_fids))) + " individuals because of case with prior abnormal ")
    pd.DataFrame({"FINNGENID":list(set(case_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_case-abnorm_fids.csv", sep=",")
    logging.info("Removed " + str(len(set(case_remove_fids))) + " individuals because of controls with prior abnormal or last abnormal if no ctrl abnormal filtering")
    pd.DataFrame({"FINNGENID":list(set(ctrl_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_ctrl-abnorm_fids.csv", sep=",")
    logging.info("Removed " + str(len(set(diag_remove_fids))) + " individuals because of diagnosis exclusions")
    pd.DataFrame({"FINNGENID":list(set(diag_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_ctrl-abnorm_fids.csv", sep=",")

    new_data.select(["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"]).write_csv(args.res_dir + out_file_name + ".csv", separator=",")
    labels.write_csv(args.res_dir + out_file_name + "_labels.csv", separator=",")
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())