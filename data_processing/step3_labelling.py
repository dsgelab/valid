# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, get_datetime, make_dir, init_logging, Timer, logging_print, read_file
from labeling_utils_new import get_lab_data, log_print_n, label_cases_and_controls,  remove_age_outliers
from diag_utils import relabel_data_diag
from processing_utils import add_set

# Standard stuff
import numpy as np
import pandas as pd
import polars as pl
# Time stuff
from datetime import datetime
from dateutil.relativedelta import relativedelta
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument("--data_path_full", type=str, help="Path to data.", default="/home/ivm/valid/data/processed_data/step1_clean/")
    parser.add_argument("--fg_ver", type=str, default="R12", help="Which FinnGen release [options: R12 and R13]")

    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--version", type=str, help="Versioning of test set for shorter file names.", required=True)

    # Settings for data-processing
    parser.add_argument("--abnorm_type", type=str, default="", help="[Options for eGFR: age, KDIGO-strict, KDIGO-soft for HbA1c: strong and soft.]. age: \
                        egfr abnormality based on age. KDIGO-stric: <60 for all. KDIGO-soft: <60 but with 60-65 allowed in between abnormal without disrupting the count.\
                        for HbA1c: strong considers values >47 abnormal and soft considers values >42 abnormal.")
    
    parser.add_argument("--start_pred_date", type=str, help="Start of prediction period [As string %Y-%m-%d].", default="2023-06-01")
    parser.add_argument("--end_pred_date", type=str, help="End of prediction period [As string %Y-%m-%d].", default="2023-12-31")
    parser.add_argument("--months_buffer", type=int, help="Minimum number months before prediction start to be removed.", default=6)
    parser.add_argument("--months_buffer_abnorm", type=int, help="Months to average before baseline, which if abnormal -> removed.", default=3)
    parser.add_argument("--min_age", type=float, help="Minimum age at prediction time", default=30)
    parser.add_argument("--max_age", type=float, help="Maximum age at prediction time", default=70)
    parser.add_argument("--test_pct", type=float, help="Percentage of test data.", default=0.2)
    parser.add_argument("--valid_pct", type=float, help="Percentage of validation data.", default=0.2)

    args = parser.parse_args()

    return(args)

def get_hbb_indvs(fg_ver="R13",
                          end_date=datetime(2023, 1, 1)) -> pl.Series:
    """Selecting individuals in FinnGen that are in Helsinki Biobank, not dead before end date."""

    # Read in the minimum data file
    if fg_ver == "R12" or fg_ver == "r12":
        minimum_file_name = "/finngen/library-red/finngen_R12/phenotype_1.0/data/finngen_R12_minimum_extended_1.0.txt.gz"
    elif fg_ver == "R13" or fg_ver == "r13":        
        minimum_file_name = "/finngen/library-red/finngen_R13/phenotype_1.0/data/finngen_R13_minimum_extended_1.0.txt.gz"
    min_data = pl.read_csv(minimum_file_name, 
                           separator="\t",
                           columns=["FINNGENID",  "COHORT"])
    # Filtering
    select_fids = (min_data
                   .filter(
                       # In Helsinki Biobank
                       (pl.col.COHORT == "HELSINKI BIOBANK") 
                   )
                   .get_column("FINNGENID")
    )
    return(select_fids)


def get_extra_file_descr(start_pred_date: pl.Date,
                         end_pred_date: pl.Date,
                         months_buffer: int,
                         test_version: str)-> str:
    extra = "test" + test_version
    if start_pred_date == datetime(2022, 6, 1) and end_pred_date == datetime(2022, 12, 31):
        extra += "_end2022"
    elif start_pred_date == datetime(2022, 1, 1) and end_pred_date == datetime(2022, 12, 31):
        extra += "_2022"
    elif start_pred_date == datetime(2022, 1, 1) and end_pred_date == datetime(2024, 12, 31):
        extra += "_2022t2024"
    else:
        raise("Please description of this prediction time period to the function 'get_extra_file_descr'.")
    if months_buffer != 0:
        extra += "_w" + str(months_buffer)
    return(extra)


if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    timer = Timer()
    args = get_parser_arguments()

    make_dir(args.res_dir)

    init_logging(args.res_dir, args.lab_name, logger, args)
    removed_ids_path = args.res_dir + "logs/removed_fids/" + args.lab_name + "/" 
    make_dir(removed_ids_path)

    end_pred_date = datetime.strptime(args.end_pred_date, "%Y-%m-%d")
    start_pred_date = datetime.strptime(args.start_pred_date, "%Y-%m-%d")
    base_date = start_pred_date - relativedelta(months=args.months_buffer)
    start_noabnorm_date = base_date - relativedelta(months=args.months_buffer_abnorm)
    
    # Setting up logging
    extra_descr = get_extra_file_descr(start_pred_date, 
                                       end_pred_date, 
                                       args.months_buffer, 
                                       args.version)
    out_file_name = args.data_path_full.split("/")[-1].split(".")[0] + "_" + extra_descr + "_" + get_date() 
    logging_print("Saving files to: "+out_file_name)
    with open(args.res_dir + out_file_name + "_call.txt", "w") as f:
        f.write("Time: " + get_datetime() + '\n--'.join(f'{k}={v}' for k, v in vars(args).items()))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = read_file(args.data_path_full)
    print(data) 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Cases and controls                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    labels = label_cases_and_controls(data=data,
                                      start_pred_date=start_pred_date,
                                      end_pred_date=end_pred_date,
                                      lab_name=args.lab_name,
                                      abnorm_type=args.abnorm_type,
                                      removed_ids_path=removed_ids_path,
                                      out_file_name=out_file_name)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Age                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    labels = remove_age_outliers(labels=labels,
                                 base_date=base_date,
                                 fg_ver=args.fg_ver,
                                 min_age=args.min_age,
                                 max_age=args.max_age,
                                 removed_ids_path=removed_ids_path,
                                 out_file_name=out_file_name)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Selecting relevant individuals                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Selecting in HBB, correct age at end, not dead, and BMI >= 18.5
    select_fids = get_hbb_indvs(fg_ver=args.fg_ver, end_date=end_pred_date)
    n_nothbb = labels.filter(~pl.col("FINNGENID").is_in(select_fids)).height
    test_labels = labels.filter(pl.col("FINNGENID").is_in(select_fids))
    test_labels = test_labels.with_columns(pl.lit(2).alias("SET"))
    other_labels = labels.filter(~pl.col.FINNGENID.is_in(select_fids))
    logging_print(str(n_nothbb) + " individuals not in HBB.")
    log_print_n(test_labels, "HBB")
    log_print_n(other_labels, "TrainValid")

    other_labels = add_set(other_labels, test_pct=args.test_pct, valid_pct=args.valid_pct)
    log_print_n(other_labels.filter(pl.col.SET==1), "Valid")
    log_print_n(other_labels.filter(pl.col.SET==0), "Train")

    labels = pl.concat([other_labels, test_labels])
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Data before start                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = data.filter((pl.col.DATE<base_date)&(pl.col.FINNGENID.is_in(labels["FINNGENID"])))

    (data
     .select(["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"])
     .write_parquet(args.res_dir + out_file_name + ".parquet"))
    labels.write_parquet(args.res_dir + out_file_name + "_labels.parquet")
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
