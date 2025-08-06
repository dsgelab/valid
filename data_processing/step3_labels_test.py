# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, get_datetime, make_dir, init_logging, Timer, logging_print
from labeling_utils import get_lab_data, log_print_n, label_cases_and_controls, remove_prior_diags, remove_only_icd_or_atc_diags, remove_buffer_abnorm, remove_last_value_abnormal, remove_age_outliers, remove_other_exclusion, get_low_bmi_indvs
from diag_utils import relabel_data_diag

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
    parser.add_argument("--diags_path", type=str, help="Path to data.", default="/home/ivm/valid/data/processed_data/step2_diags/")
    parser.add_argument("--diags_file_name", type=str, help="File name of diagnosis file.")
    parser.add_argument("--exclude_file_name", type=str, help="File name of excluded individuals.")

    parser.add_argument("--fg_ver", type=str, default="R12", help="Which FinnGen release [options: R12 and R13]")

    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--version", type=str, help="Versioning of test set for shorter file names.", required=True)

    # Settings for data-processing
    parser.add_argument("--exclude_diags", type=int, default=0, help="Whether to exclude individuals with prior diags or not. 0 = no, 1 = yes")
    parser.add_argument("--data_diag_excl", type=int, default=0, help="Whether to remove individuals with only a data-based diagnosis prior to the start.")
    parser.add_argument("--shorten_diag_pred", type=int, default=0, help="Whether to relabel all diags based on a different window than the exclusion one.")
    parser.add_argument("--diff_days", type=int, default=None, help="Minimum number of days between measurements needed for diagnosis. Needed when relabeling data-based diagnoses to after start-date.")

    parser.add_argument("--remove_low_bmi", type=int, help="Whether to remove individuals with a BMI<18.5", default=0)
    parser.add_argument("--no_prior_diag_start", type=int, help="Whether to only consider cases who has first abnormal after the baseline. Removing those with first data-base abnormal before.", default=0)
    parser.add_argument("--abnorm_type", type=str, required=True, help="[Options for eGFR: age, KDIGO-strict, KDIGO-soft for HbA1c: strong and soft.]. age: \
                        egfr abnormality based on age. KDIGO-stric: <60 for all. KDIGO-soft: <60 but with 60-65 allowed in between abnormal without disrupting the count.\
                        for HbA1c: strong considers values >47 abnormal and soft considers values >42 abnormal.")
    
    parser.add_argument("--start_pred_date", type=str, help="Start of prediction period [As string %Y-%m-%d].", default="2023-06-01")
    parser.add_argument("--end_pred_date", type=str, help="End of prediction period [As string %Y-%m-%d].", default="2023-12-31")
    parser.add_argument("--months_buffer", type=int, help="Minimum number months before prediction start to be removed.", default=6)
    parser.add_argument("--months_buffer_abnorm", type=int, help="Months to average before baseline, which if abnormal -> removed.", default=3)
    parser.add_argument("--min_age", type=float, help="Minimum age at prediction time", default=30)
    parser.add_argument("--max_age", type=float, help="Maximum age at prediction time", default=70)

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

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    timer = Timer()
    args = get_parser_arguments()
    out_file_name = args.diags_file_name.split(".")[0] + "_test_" + args.version + "_" + get_date() 

    make_dir(args.res_dir)
    with open(args.res_dir + out_file_name + "_call.txt", "w") as f:
        f.write("Time: " + get_datetime() + '\n--'.join(f'{k}={v}' for k, v in vars(args).items()))

    init_logging(args.res_dir, args.lab_name, logger, args)
    removed_ids_path = args.res_dir + "logs/removed_fids/" + args.lab_name + "/" 
    make_dir(removed_ids_path)
    logging.info("Saving files to: "+out_file_name)

    end_pred_date = datetime.strptime(args.end_pred_date, "%Y-%m-%d")
    start_pred_date = datetime.strptime(args.start_pred_date, "%Y-%m-%d")
    base_date = start_pred_date - relativedelta(months=args.months_buffer)
    start_noabnorm_date = base_date - relativedelta(months=args.months_buffer_abnorm)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = get_lab_data(data_path=args.data_path_full, 
                        diags_path=args.diags_path+args.diags_file_name)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Remove "wrong" diagnoses data                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = remove_prior_diags(data=data,
                              start_pred_date=start_pred_date,
                              data_diag_excl=args.data_diag_excl,
                              removed_ids_path=removed_ids_path,
                              out_file_name=out_file_name)
    if args.data_diag_excl == 0:
        data = relabel_data_diag(data=data,
                                 start_pred_date=start_pred_date,
                                 diff_days=args.diff_days)
    if args.shorten_diag_pred == 1:
        data = relabel_data_diag(data=data,
                                 start_pred_date=end_pred_date,
                                 diff_days=args.diff_days)
    if args.lab_name == "egfr":
        data = remove_only_icd_or_atc_diags(data=data,
                                            removed_ids_path=removed_ids_path,
                                            out_file_name=out_file_name,
                                            lab_name=args.lab_name)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Remove prior abnormal                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.monthss_buffer_abnorm>0:
        data = remove_buffer_abnorm(data=data,
                                    start_noabnorm_date=start_noabnorm_date,
                                    base_date=base_date,
                                    lab_name=args.lab_name,
                                    abnorm_type=args.abnorm_type,
                                    removed_ids_path=removed_ids_path,
                                    out_file_name=out_file_name)
        
    if args.no_prior_diag_start == 1:
        data = remove_last_value_abnormal(data=data,
                                          removed_ids_path=removed_ids_path,
                                          out_file_name=out_file_name)
        
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
    #                 Diag exclusions                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if args.exclude_diags == 1:
        labels = remove_other_exclusion(labels=labels,
                                        diags_path=args.diags_path,
                                        exclude_file_name=args.exclude_file_name,
                                        base_date=base_date,
                                        removed_ids_path=removed_ids_path,
                                        out_file_name=out_file_name)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Selecting relevant individuals                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Selecting in HBB, correct age at end, not dead, and BMI >= 18.5
    select_fids = get_hbb_indvs(fg_ver=args.fg_ver, end_date=end_pred_date)
    n_nothbb = labels.filter(~pl.col("FINNGENID").is_in(select_fids)).height
    labels = labels.filter(pl.col("FINNGENID").is_in(select_fids))
    logging_print("Removed " + str(n_nothbb) + " individuals not in HBB.")
    log_print_n(labels, "HBB")

    if args.lab_name=="egfr" and args.remove_low_bmi == 1:
        select_fids = get_low_bmi_indvs(fg_ver=args.fg_ver) 
        n_lowbmi = labels.filter(~pl.col("FINNGENID").is_in(select_fids)).height
        logging_print("Removed " + str(n_lowbmi) + " individuals with BMI<18.5.")
        log_print_n(labels, "BMI")


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
