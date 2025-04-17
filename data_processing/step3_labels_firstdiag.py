# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, make_dir, init_logging, Timer, read_file, logging_print
from processing_utils import add_measure_counts, add_set
from labeling_utils import get_lab_data, get_cases, get_controls, sample_controls, add_pred_helpers, remove_future_data
# Standard stuff
import numpy as np
import pandas as pd
import polars as pl
# Time stuff
from datetime import datetime
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse

def get_extra_file_descr(no_abnorm: int,
                         normal_before_diag: int,
                         sample_ctrls: int,
                         start_year: int) -> str:
    extra = ""
    if start_year > 2013:
        extra += "start" + str(int(args.start_year))
    if no_abnorm == 1:
        extra = "-noabnorm"
    if normal_before_diag:
        extra += "-normbefore"     
    if sample_ctrls == 1:
        extra += "-ctrlsample"

    return(extra + "_")

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument("--data_path_full", type=str, help="Path to data.", default="/home/ivm/valid/data/processed_data/step1_clean/")
    parser.add_argument("--diags_path", type=str, help="Path to data.", default="/home/ivm/valid/data/processed_data/step2_diags/")
    parser.add_argument("--diags_file_name", type=str, help="File name of diagnosis file.")
    parser.add_argument("--exclude_file_name", type=str, help="File name of excluded individuals.")
    parser.add_argument("--exclude_diags", type=int, default=0, help="Whether to exclude individuals with prior diags or not. 0 = no, 1 = yes")
    parser.add_argument("--test_path_start", type=str, default="", help="Path to individuals that are part of the pre-defined test set.")
    parser.add_argument("--fg_ver", type=str, default="R12", help="Which FinnGen release [options: R12 and R13]")
    parser.add_argument("--end_pred_date", type=str, help="End of prediction period [As string %Y-%m-%d].", default="2023-12-31")

    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)

    # Settings for data-processing
    parser.add_argument("--test_pct", type=float, help="Percentage of test data.", default=0.2)
    parser.add_argument("--valid_pct", type=float, help="Percentage of validation data.", default=0.2)
    parser.add_argument("--no_abnorm", type=float, help="Whether to create super controls where all prior data is normal.", default=1)
    parser.add_argument("--normal_before_diag", type=float, help="Whether cases need at least one recorded normal before diagnosis.", default=0)
    parser.add_argument("--sample_ctrls", type=float, help="Whether cases need at least one recorded normal before diagnosis.", default=0)
    parser.add_argument("--start_year", type=float, help="Minimum number of years with at least one measurement to be included", default=2013)

    parser.add_argument("--min_n_years", type=float, help="Minimum number of years with at least one measurement to be included", default=1)
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
    extra = get_extra_file_descr(args.no_abnorm, args.normal_before_diag, args.sample_ctrls, args.start_year)
    out_file_name = args.diags_file_name.split(".")[0] + "_" + extra + get_date() 
        
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.lab_name, logger, args)
    removed_ids_path = args.res_dir + "logs/removed_fids/" + args.lab_name + "/" 
    make_dir(removed_ids_path)
    logging.info("Saving files to: "+out_file_name)
    end_pred_date = datetime.strptime(args.end_pred_date, "%Y-%m-%d")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = get_lab_data(data_path=args.data_path_full, 
                        diags_path=args.diags_path+args.diags_file_name)    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Remove future data                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Future lab data
    data = remove_future_data(data, end_pred_date, strict=True)
    ### Processing
    if args.min_n_years > 1:
        data = add_measure_counts(data)
        nyear_fids = set(data.filter(pl.col("N_YEAR")==args.min_n_years)["FINNGENID"])

        data = data.filter(~pl.col("FINNGENID").is_in(nyear_fids))

        logging.info("Removed " + str(len(nyear_fids)) + " individuals with less than two years data")
        pd.DataFrame({"FINNGENID":list(nyear_fids)}).to_csv(removed_ids_path + out_file_name + "_reason_nyears.csv", sep=",")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Removing test set individuals                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.test_path_start != "":
        test_labels = read_file(args.test_path_start + "_labels.parquet")
        test_labels = test_labels.with_columns(SET=2)
        test_data = read_file(args.test_path_start + ".parquet")
        data = data.filter(~pl.col("FINNGENID").is_in(test_labels["FINNGENID"]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Cases and controls                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    cases, case_remove_fids = get_cases(data=data, 
                                        no_abnorm=args.no_abnorm, 
                                        months_buffer=args.months_buffer, 
                                        normal_before_diag=args.normal_before_diag)
    controls, ctrl_remove_fids = get_controls(data=data, 
                                              no_abnorm=args.no_abnorm, 
                                              months_buffer=args.months_buffer)
    new_data = pl.concat([cases, controls.select(cases.columns)]) 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Diag exclusions                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if args.exclude_diags == 1:
        exclusion_data = read_file(args.diags_path+args.exclude_file_name)
        diag_remove_fids = (exclusion_data
                            .join(new_data.select(["FINNGENID", "START_DATE"]), on="FINNGENID", how="left")
                            .filter(pl.col("EXCL_DATE") < pl.col("START_DATE"))
                            .get_column("FINNGENID"))
        new_data = new_data.filter(~pl.col("FINNGENID").is_in(diag_remove_fids))
        
        logging.info("Removed " + str(len(set(diag_remove_fids))) + " individuals because of diagnosis exclusions.")
        pd.DataFrame({"FINNGENID":list(set(diag_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_diag-exclusion_fids.csv", sep=",")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Control sampling                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.sample_ctrls == 1:
        controls = sample_controls(new_data.filter(pl.col("y_DIAG")==1), 
                                   new_data.filter(pl.col("y_DIAG")==0), 
                                   args.months_buffer)
        new_data = pl.concat([new_data.filter(pl.col("y_DIAG")==1), 
                              controls.select(new_data.columns)]) 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Add age and predicted value y_MEAN                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    new_data = add_pred_helpers(new_data, fg_ver=args.fg_ver)
    # It can happen because of ICD-code diagnosis and no after measurements
    new_data = new_data.filter(~pl.col.y_MEAN.is_null())

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Year                                                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    year_remove_fids = set(new_data.filter(
                                pl.col("PRED_DATE").dt.year()<(args.start_year)
                            ).get_column("FINNGENID")
                        ) 
    new_data = new_data.filter(~pl.col("FINNGENID").is_in(year_remove_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Age                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    age_remove_fids = set(new_data.filter(
                            (pl.col("END_AGE")<args.min_age)|(pl.col("END_AGE")>args.max_age)
                             ).get_column("FINNGENID")
                        )
    new_data = new_data.filter(~pl.col("FINNGENID").is_in(age_remove_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Labels                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    labels = (new_data
              .select(["FINNGENID", "y_DIAG", "END_AGE", "SEX","y_MEAN", "START_DATE", "PRED_DATE"])
              .rename({"END_AGE": "EVENT_AGE"})
              .unique(subset=["FINNGENID", "y_DIAG"])
    )
    labels = add_set(labels, test_pct=args.test_pct, valid_pct=args.valid_pct)
    if args.test_path_start != "":
        labels = pl.concat([labels, test_labels])
    logging_print(f"N rows total data {len(new_data)} N rows labels {len(labels)}  N indvs {len(set(labels["FINNGENID"]))}  N cases {sum(labels["y_DIAG"])} N controls {sum(labels["y_DIAG"]==0)} pct cases {round(sum(labels["y_DIAG"])/len(labels)*100,2)}%")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Data before start                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    new_data = new_data.filter(pl.col("DATE") < pl.col("START_DATE"))
    new_data = new_data.select(["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"])
    if args.test_path_start != "":
        new_data = pl.concat([new_data, test_data])

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info("Removed " + str(len(age_remove_fids)) + " individuals with age < " + str(args.min_age) + " or > " + str(args.max_age))
    pd.DataFrame({"FINNGENID":list(set(age_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_age.csv", sep=",")
    logging.info("Removed " + str(len(year_remove_fids)) + " individuals with time of prediction < " + str(args.start_year))
    pd.DataFrame({"FINNGENID":list(set(year_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_year.csv", sep=",")
    logging.info("Removed " + str(len(set(case_remove_fids))) + " individuals because of case with prior abnormal ")
    pd.DataFrame({"FINNGENID":list(set(case_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_case-abnorm_fids.csv", sep=",")
    logging.info("Removed " + str(len(set(ctrl_remove_fids))) + " individuals because of controls with prior abnormal or last abnormal if no ctrl abnormal filtering")
    pd.DataFrame({"FINNGENID":list(set(ctrl_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_ctrl-abnorm_fids.csv", sep=",")

    new_data.write_parquet(args.res_dir + out_file_name + ".parquet")
    labels.write_parquet(args.res_dir + out_file_name + "_labels.parquet")
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
