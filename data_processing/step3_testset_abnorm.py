# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, get_datetime, make_dir, init_logging, Timer, read_file, logging_print
from labeling_abnorm_utils import get_lab_data, add_ages, remove_future_diags, log_print_n
from processing_utils import get_abnorm_func_based_on_name
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
    parser.add_argument("--exclude_diags", type=int, default=0, help="Whether to exclude individuals with prior diags or not. 0 = no, 1 = yes")
    parser.add_argument("--fg_ver", type=str, default="R12", help="Which FinnGen release [options: R12 and R13]")

    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)

    # Settings for data-processing
    parser.add_argument("--no_prior_diag_start", type=int, help="Whether to only consider cases who has first abnormal after the baseline. Removing those with first data-base abnormal before.", default=0)

    parser.add_argument("--start_pred_date", type=str, help="Start of prediction period [As string %Y-%m-%d].", default="2023-06-01")
    parser.add_argument("--end_pred_date", type=str, help="End of prediction period [As string %Y-%m-%d].", default="2023-12-31")
    parser.add_argument("--months_buffer", type=int, help="Minimum number months before prediction start to be removed.", default=6)
    parser.add_argument("--months_buffer_abnorm", type=int, help="Months to average before baseline, which if abnormal -> removed.", default=3)
    parser.add_argument("--min_age", type=float, help="Minimum age at prediction time", default=30)
    parser.add_argument("--max_age", type=float, help="Maximum age at prediction time", default=70)

    args = parser.parse_args()

    return(args)

def get_egfr_select_fids(fg_ver="R13",
                         end_date=datetime(2023, 1, 1)) -> pl.Series:
    """Selecting individuals in FinnGen that are in Helsinki Biobank, not dead before 2023,
    and aged between 30 and 70 at prediction time (or what is set as min and max). As well as, 
    BMI not recorded or BMI >= 18.5 (low might make eGFR unreliable)."""

    # Read in the minimum data file
    if fg_ver == "R12" or fg_ver == "r12":
        minimum_file_name = "/finngen/library-red/finngen_R12/phenotype_1.0/data/finngen_R12_minimum_extended_1.0.txt.gz"
    elif fg_ver == "R13" or fg_ver == "r13":        
        minimum_file_name = "/finngen/library-red/finngen_R13/phenotype_1.0/data/finngen_R12_minimum_extended_1.0.txt.gz"
    min_data = pl.read_csv(minimum_file_name, 
                           separator="\t",
                           columns=["FINNGENID", "APPROX_BIRTH_DATE", "COHORT", "DEATH_APPROX_EVENT_DAY", "BMI"])
    min_data = min_data.with_columns([
                pl.col.APPROX_BIRTH_DATE.cast(pl.Utf8).str.to_date(strict=False).alias("APPROX_BIRTH_DATE"),
                pl.col.DEATH_APPROX_EVENT_DAY.cast(pl.Utf8).str.to_date(strict=False).alias("DEATH_APPROX_EVENT_DAY"),
                pl.col.BMI.cast(pl.Float64, strict=False).alias("BMI")
    ])
    # Filtering
    select_fids = (min_data
                   .filter(
                       # In Helsinki Biobank
                       (pl.col.COHORT == "HELSINKI BIOBANK") &
                       # Did not die 
                       ((pl.col.DEATH_APPROX_EVENT_DAY.is_null())|(pl.col.DEATH_APPROX_EVENT_DAY<end_date)) &
                       # BMI not recorded or BMI >= 18.5 (low might make eGFR unreliable)
                       ((pl.col.BMI.is_null()) | (pl.col.BMI>=18.5)) 
                   )
                   .get_column("FINNGENID")
    )
    return(select_fids)

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
                           columns=["FINNGENID", "APPROX_BIRTH_DATE", "COHORT", "DEATH_APPROX_EVENT_DAY", "BMI"])
    min_data = min_data.with_columns([
                pl.col.APPROX_BIRTH_DATE.cast(pl.Utf8).str.to_date(strict=False).alias("APPROX_BIRTH_DATE"),
                pl.col.DEATH_APPROX_EVENT_DAY.cast(pl.Utf8).str.to_date(strict=False).alias("DEATH_APPROX_EVENT_DAY"),
                pl.col.BMI.cast(pl.Float64, strict=False).alias("BMI")
    ])
    # Filtering
    select_fids = (min_data
                   .filter(
                       # In Helsinki Biobank
                       (pl.col.COHORT == "HELSINKI BIOBANK") &
                       # Did not die 
                       ((pl.col.DEATH_APPROX_EVENT_DAY.is_null())|(pl.col.DEATH_APPROX_EVENT_DAY<end_date)) 
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
    out_file_name = args.diags_file_name.split(".")[0] + "_test_" + get_date() 
        
    make_dir(args.res_dir)
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
    #                 Remove future data                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data_remove = remove_future_diags(data, start_pred_date, strict=True)
    prior_diags_fids = data_remove.filter(~pl.col.FIRST_DIAG_DATE.is_null())["FINNGENID"].unique()
    data = data.filter(~pl.col.FINNGENID.is_in(prior_diags_fids))

    logging_print("Removed " + str(len(set(prior_diags_fids))) + " individuals because of prior CKD diagnosis.")
    pd.DataFrame({"FINNGENID":list(set(prior_diags_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_prior-ckd_fids.csv", sep=",")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Remove ICD-or-ATC-but-not-data-diagnosed                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    other_diags_fids = data.filter((pl.col.DATA_DIAG_DATE.is_null()&(~pl.col.FIRST_DIAG_DATE.is_null())))["FINNGENID"].unique()
    data = data.filter(~pl.col.FINNGENID.is_in(other_diags_fids))

    logging_print("Removed " + str(len(set(other_diags_fids))) + " individuals because of other diagosis but not data.")
    pd.DataFrame({"FINNGENID":list(set(other_diags_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_prior-other-diags_fids.csv", sep=",")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Remove buffer abnormal                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.months_buffer_abnorm>0:
        abnorm_data = (data
                       .filter((pl.col.DATE >= start_noabnorm_date)&(pl.col.DATE < base_date))
                       .with_columns(pl.col.VALUE.mean().over("FINNGENID").alias("MEAN_PRIOR"))
                       .select("FINNGENID", "MEAN_PRIOR")
        )
        abnorm_data = get_abnorm_func_based_on_name(args.lab_name)(abnorm_data, "MEAN_PRIOR")
        abnorm_fids = abnorm_data.filter(pl.col.ABNORM_CUSTOM>0).get_column("FINNGENID").unique()
        data = data.filter(~pl.col.FINNGENID.is_in(abnorm_fids))
        logging_print("Removed " + str(len(set(abnorm_fids))) + " individuals because of prior mean abnormal.")
        pd.DataFrame({"FINNGENID":list(set(abnorm_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_prior-abnorm_fids.csv", sep=",")
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Cases and controls                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.no_prior_diag_start == 1:
        # Diagnosis is inside prediction period
        # But first abnormal is not before baseline date
        last_val_abnormal_fids = (data.filter(pl.col.DATE<base_date)
                                      .filter(pl.col.ABNORM_CUSTOM!=0.5)
                                      .filter(pl.col.DATE==pl.col.DATE.max().over("FINNGENID"))
                                      .filter(pl.col.ABNORM_CUSTOM==1)
                                      .get_column("FINNGENID")
                                      .unique()
                   )
        logging_print("Removed " + str(len(set(last_val_abnormal_fids))) + " individuals because of last value abnormal.")
        pd.DataFrame({"FINNGENID":list(set(last_val_abnormal_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_last-abnormal_fids.csv", sep=",")
        data = data.filter(~pl.col.FINNGENID.is_in(last_val_abnormal_fids))
        
    labels = (data.filter((pl.col.DATE>=start_pred_date)&(pl.col.DATE<=end_pred_date))
                .with_columns(
                    pl.col.VALUE.mean().over("FINNGENID").alias("y_MEAN"),
                    pl.col.VALUE.min().over("FINNGENID").alias("y_MIN"),
                    pl.col.VALUE.get(pl.col.DATE.arg_min()).over("FINNGENID").alias("y_NEXT"),
                    # Cases should have the diagnosis date inside the prediction period
                    # Will not have first abnormal here, if heappened earlier but no diag yet.
                    # Or also the first abnormal is enough here
                    pl.when(((pl.col.DATA_FIRST_DIAG_ABNORM_DATE>=start_pred_date)&(pl.col.DATA_FIRST_DIAG_ABNORM_DATE<=end_pred_date)) | \
                            ((pl.col.DATA_DIAG_DATE>=start_pred_date)&(pl.col.DATA_DIAG_DATE<=end_pred_date)))              
                      .then(1).otherwise(0).alias("y_DIAG")

                )
    )
    labels = get_abnorm_func_based_on_name(args.lab_name)(labels, "y_MEAN").rename({"ABNORM_CUSTOM": "y_MEAN_ABNORM"})
    labels = get_abnorm_func_based_on_name(args.lab_name)(labels, "y_MIN").rename({"ABNORM_CUSTOM": "y_MIN_ABNORM"})
    labels = get_abnorm_func_based_on_name(args.lab_name)(labels, "y_NEXT").rename({"ABNORM_CUSTOM": "y_NEXT_ABNORM"})
    labels = labels.select("FINNGENID", "SEX", "y_MEAN", "y_MEAN_ABNORM", "y_MIN", "y_MIN_ABNORM", "y_NEXT", "y_NEXT_ABNORM", "y_DIAG").unique()
    
    no_labels_data_fids = data.filter(~pl.col.FINNGENID.is_in(labels["FINNGENID"]))["FINNGENID"].unique()
    logging_print("Removed " + str(len(set(no_labels_data_fids))) + " individuals because of no measurement in the time.")
    pd.DataFrame({"FINNGENID":list(set(no_labels_data_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_no-measure_fids.csv", sep=",")
    
    log_print_n(labels, "Start")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Age                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    labels = add_ages(labels, base_date, fg_ver=args.fg_ver)
    age_remove_fids = set(labels
                          .filter((pl.col("EVENT_AGE")<args.min_age)|(pl.col("EVENT_AGE")>args.max_age))
                          .get_column("FINNGENID")
    )
    logging_print("Removed " + str(len(set(age_remove_fids))) + " individuals because of age.")
    pd.DataFrame({"FINNGENID":list(set(age_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_age_fids.csv", sep=",")
    
    labels = labels.filter(~pl.col("FINNGENID").is_in(age_remove_fids))
    log_print_n(labels, "Age")
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Diag exclusions                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if args.exclude_diags == 1:
        exclusion_data = read_file(args.diags_path+args.exclude_file_name)
        # We do not want any of the exclusion individuals in the test set
        excl_remove_fids = (exclusion_data
                            .filter((pl.col("EXCL_DATE") < base_date)&(pl.col.FINNGENID.is_in(labels["FINNGENID"])))
                            .get_column("FINNGENID"))
        labels = labels.filter(~pl.col("FINNGENID").is_in(excl_remove_fids))

        logging_print("Removed " + str(len(set(excl_remove_fids))) + " individuals because of diagnosis exclusions.")
        pd.DataFrame({"FINNGENID":list(set(excl_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_diag-exclusion_fids.csv", sep=",")

    log_print_n(labels, "Exclusion")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Selecting relevant individuals                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Selecting in HBB, correct age at end, not dead, and BMI >= 18.5
    if args.lab_name=="hba1c" or args.lab_name=="tsh":
        select_fids = get_hbb_indvs(fg_ver=args.fg_ver, end_date=end_pred_date)
    if args.lab_name=="egfr":
        select_fids = get_egfr_select_fids(fg_ver=args.fg_ver, end_date=end_pred_date) 
    n_nothbb = labels.filter(~pl.col("FINNGENID").is_in(select_fids)).height
    labels = labels.filter(pl.col("FINNGENID").is_in(select_fids))
    if args.lab_name=="hba1c":
        logging_print("Removed " + str(n_nothbb) + " individuals not in HBB or dead.")
    if args.lab_name=="egfr":
        logging_print("Removed " + str(n_nothbb) + " individuals not in HBB, dead or BMI<18.5.")

    log_print_n(labels, "HBB")

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
