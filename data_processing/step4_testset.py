# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, make_dir, init_logging, Timer, read_file, logging_print
from labeling_utils import get_lab_data, get_cases, get_controls
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


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument("--data_path", type=str, help="Path to data.", default="/home/ivm/valid/data/processed_data/step3_meta/")
    parser.add_argument("--file_name", type=str, help="Path to data.")
    parser.add_argument("--exclusion_path", type=str, help="Path to exclusions, atm excluding all FGIDs in there.")
    parser.add_argument("--exclude_diags", type=int, default=0, help="Whether to exclude individuals with prior diags or not. 0 = no, 1 = yes")
    parser.add_argument("--fg_ver", type=str, default="R12", help="Which FinnGen release [options: R12 and R13]")

    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)

    # Settings for data-processing
    parser.add_argument("--sample_ctrls", type=float, help="Whether cases need at least one recorded normal before diagnosis.", default=0)
    parser.add_argument("--start_pred_date", type=str, help="Start of prediction period [As string %Y-%m-%d].", default="2023-06-01")
    parser.add_argument("--end_pred_date", type=str, help="End of prediction period [As string %Y-%m-%d].", default="2023-12-31")
    parser.add_argument("--months_buffer", type=int, help="Minimum number months before prediction start to be removed.", default=6)
    parser.add_argument("--min_age", type=float, help="Minimum age at prediction time", default=30)
    parser.add_argument("--max_age", type=float, help="Maximum age at prediction time", default=70)

    args = parser.parse_args()

    return(args)

def get_egfr_select_fids(fg_ver="R13",
                         end_date=datetime(2023, 1, 1),
                         min_age=30,
                         max_age=70) -> pl.Series:
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
                       (pl.col.DEATH_APPROX_EVENT_DAY.is_null()) &
                       # Aged between 30 and 70 at prediction time (or what is set as min and max)
                       (pl.col.APPROX_BIRTH_DATE.dt.year() <= end_date.year-min_age) &
                       (pl.col.APPROX_BIRTH_DATE.dt.year() >= end_date.year-max_age) &
                       # BMI not recorded or BMI >= 18.5 (low might make eGFR unreliable)
                       ((pl.col.BMI.is_null()) | (pl.col.BMI>=18.5)) 
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
    out_file_name = args.file_name + get_date() 
        
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.lab_name, logger, args)
    removed_ids_path = args.res_dir + "logs/removed_fids/" + args.lab_name + "/" 
    make_dir(removed_ids_path)
    logging.info("Saving files to: "+out_file_name)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Selecting relevant individuals                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Selecting in HBB, correct age at end, not dead, and BMI >= 18.5
    select_fids = get_egfr_select_fids(fg_ver=args.fg_ver,
                                       end_date=datetime.strptime(args.end_pred_date, "%Y-%m-%d"),
                                       min_age=args.min_age,
                                       max_age=args.max_age)
    data = get_lab_data(data_path=args.data_path, file_name=args.file_name)
    min_removed_fids = set(data.filter(~pl.col("FINNGENID").is_in(select_fids)).get_column("FINNGENID"))
    data = data.filter(pl.col("FINNGENID").is_in(select_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Cases and controls                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    cases, case_remove_fids = get_cases(data=data, 
                                        no_abnormal_cases=0, 
                                        months_buffer=args.months_buffer, 
                                        normal_before_diag=0)
    controls, ctrl_remove_fids = get_controls(data=data, 
                                              no_abnormal_ctrls=0, 
                                              months_buffer=args.months_buffer)
    new_data = pl.concat([cases, controls.select(cases.columns)]) 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 End before start of observation                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    date_remove_fids = set(new_data
                           # Careful this basically assumes that the start date
                           # cannot be after the planned prediction frame
                           # currently not the idea but might be confusing 
                           # from the coding concept
                           .filter(pl.col("PRED_DATE")<args.start_pred_date)
                           .get_column("FINNGENID")
                           )
    new_data = new_data.filter(~pl.col("FINNGENID").is_in(date_remove_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Diag exclusions                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if args.exclude_diags == 1:
        exclusion_data = read_file(args.exclusion_path)
        # We do not want any of the exclusion individuals in the test set
        diag_remove_fids = (exclusion_data
                            .join(new_data.select(["FINNGENID", "PRED_DATE"]), on="FINNGENID", how="left")
                            .filter(pl.col("EXCL_DATE") > pl.col("PRED_DATE"))
                            .get_column("FINNGENID"))
        new_data = new_data.filter(~pl.col("FINNGENID").is_in(diag_remove_fids))
        logging.info("Removed " + str(len(set(diag_remove_fids))) + " individuals because of diagnosis exclusions.")
        pd.DataFrame({"FINNGENID":list(set(diag_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_diag-exclusion_fids.csv", sep=",")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Labels                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    labels = (new_data
              .select(["FINNGENID", "y_DIAG", "END_AGE", "SEX","y_MEAN", "START_DATE", "PRED_DATE"])
              .rename({"END_AGE": "EVENT_AGE"})
              .unique(subset=["FINNGENID", "y_DIAG"]))
    logging_print(f"N rows data {len(new_data)} N rows labels {len(labels)}  N indvs {len(set(labels["FINNGENID"]))}  N cases {sum(labels["y_DIAG"])} pct cases {round(sum(labels["y_DIAG"])/len(labels)*100,2)}%")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Data before start                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    new_data = new_data.filter(pl.col("DATE") < pl.col("START_DATE"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info("Removed " + str(len(min_removed_fids)) + " individuals not in Helsinki Biobank or dead before 2023 or BMI < 18.5")
    pd.DataFrame({"FINNGENID":list(set(min_removed_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_minextended.csv", sep=",")
    logging.info("Removed " + str(len(date_remove_fids)) + " individuals with time of prediction < " + str(args.start_pred_date))
    pd.DataFrame({"FINNGENID":list(set(date_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_date.csv", sep=",")
    logging.info("Removed " + str(len(set(case_remove_fids))) + " individuals because of case with prior abnormal ")
    pd.DataFrame({"FINNGENID":list(set(case_remove_fids))}).to_csv(removed_ids_path +  out_file_name + "_reason_case-abnorm_fids.csv", sep=",")
    logging.info("Removed " + str(len(set(ctrl_remove_fids))) + " individuals because of controls with prior abnormal or last abnormal if no ctrl abnormal filtering")
    pd.DataFrame({"FINNGENID":list(set(ctrl_remove_fids))}).to_csv(removed_ids_path +  out_file_name + "_reason_ctrl-abnorm_fids.csv", sep=",")


    (new_data
     .select(["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"])
     .write_parquet(args.res_dir + out_file_name + ".parquet"))
    labels.write_parquet(args.res_dir + out_file_name + "_labels.parquet")
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
