# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, get_datetime, make_dir, init_logging, Timer, logging_print, read_file
from labeling_utils import log_print_n, label_cases_and_controls,  remove_age_outliers, get_extra_file_descr, get_bbs_indvs, add_set

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
    parser.add_argument("--data_path_full", type=str, help="Path to data.", required=True)
    parser.add_argument("--val_data_path_full", type=str, help="Path to validation data (future).", required=False)

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
    parser.add_argument("--val_start_pred_date", type=str, help="Start of validation prediction period [As string %Y-%m-%d].", default="2023-06-01")

    parser.add_argument("--end_pred_date", type=str, help="End of prediction period [As string %Y-%m-%d].", default="2023-12-31")
    parser.add_argument("--val_end_pred_date", type=str, help="End of validation prediction period [As string %Y-%m-%d].", default="2023-12-31")

    parser.add_argument("--min_per_year", type=bool, help="Take mean per year and the minimum/maximum (depending on the lab type) of that.", default=False)
    parser.add_argument("--months_buffer", type=int, help="Minimum number months before prediction start to be removed.", default=6)
    parser.add_argument("--min_age", type=float, help="Minimum age at prediction time", default=30)
    parser.add_argument("--max_age", type=float, help="Maximum age at prediction time", default=70)
    
    parser.add_argument("--n_test", type=int, help="Number of individuals in test data.", default=0)
    parser.add_argument("--valid_pct", type=float, help="Percentage of final validation data.", default=0.2)
    parser.add_argument("--finetune_valid_pct", type=float, help="Percentage of finetune-validation data. For feature selection and optuna optimization.", default=0.2)
    parser.add_argument("--test_bbs", type=str, default=["HELSINKI BIOBANK"], nargs="+", help="Cohorts that should be part of the test set i.e. HELSINKI BIOBANK")
    parser.add_argument("--future_val", type=bool, help="Use all current data for training. Validation on future data.", default=False)


    args = parser.parse_args()

    return(args)

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
    val_end_pred_date = datetime.strptime(args.val_end_pred_date, "%Y-%m-%d")
    start_pred_date = datetime.strptime(args.start_pred_date, "%Y-%m-%d")
    val_start_pred_date = datetime.strptime(args.val_start_pred_date, "%Y-%m-%d")

    base_date = start_pred_date - relativedelta(months=args.months_buffer)
    val_base_date = val_start_pred_date - relativedelta(months=args.months_buffer)
    
    # Setting up logging
    extra_descr = get_extra_file_descr(start_pred_date, 
                                       end_pred_date, 
                                       val_start_pred_date,
                                       val_end_pred_date,
                                       args.months_buffer, 
                                       args.version)
    out_file_name = args.data_path_full.split("/")[-1].split(".")[0] + "_" + extra_descr + "_" + get_date() 
    logging_print("Saving files to: "+out_file_name)
    with open(args.res_dir + out_file_name + "_call.txt", "w") as f:
        f.write("Time: " + get_datetime() + '\n--'.join(f'{k}={v}' for k, v in vars(args).items()))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    print(args.data_path_full)
    data = read_file(args.data_path_full)
    if args.future_val:
        val_data = read_file(args.val_data_path_full)
        
    #print(data) 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Cases and controls                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    labels = label_cases_and_controls(data=data,
                                      start_pred_date=start_pred_date,
                                      end_pred_date=end_pred_date,
                                      min_per_year=args.min_per_year,
                                      lab_name=args.lab_name,
                                      abnorm_type=args.abnorm_type,
                                      removed_ids_path=removed_ids_path,
                                      out_file_name=out_file_name)
    if args.future_val:
        val_labels = label_cases_and_controls(data=val_data,
                                            start_pred_date=val_start_pred_date,
                                            end_pred_date=val_end_pred_date,
                                            min_per_year=args.min_per_year,
                                            lab_name=args.lab_name,
                                            abnorm_type=args.abnorm_type,
                                            removed_ids_path=removed_ids_path + "val_",
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
    if args.future_val:
        val_labels = remove_age_outliers(labels=val_labels,
                                         base_date=val_base_date,
                                         fg_ver=args.fg_ver,
                                         min_age=args.min_age,
                                         max_age=args.max_age,
                                         removed_ids_path=removed_ids_path + "val_",
                                         out_file_name=out_file_name)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Selecting relevant individuals                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    # # # # # # # # # # #  Test set biobank  # # # # # # # # # # # # # # # #
    select_fids = get_bbs_indvs(fg_ver=args.fg_ver, bbs=args.test_bbs)
    if args.n_test==0: n_test = int(len(select_fids)*args.test_pct)
    else: n_test = args.n_test
    np.random.seed(230123)

    if not args.future_val:
        select_fids = np.random.choice(labels.filter(pl.col.FINNGENID.is_in(select_fids))["FINNGENID"].to_list(), n_test, replace=False).tolist()
        test_labels = (labels
                    .filter(pl.col("FINNGENID").is_in(select_fids))
                    .with_columns(pl.lit(2).cast(pl.Float64).alias("SET"))
                    .unique()
        )
        other_labels = labels.filter(~pl.col.FINNGENID.is_in(select_fids))
        logging_print(str(other_labels.height) + " individuals not in HBB.")
        log_print_n(test_labels, "Test")
        log_print_n(other_labels, "TrainValid")

        other_labels = add_set(other_labels, 
                               valid_pct=args.valid_pct,
                               finetune_valid_pct=args.finetune_valid_pct)

        labels = pl.concat([other_labels, test_labels])

        log_print_n(labels.filter(pl.col.SET==0.5), "Finetune Valid")
        log_print_n(labels.filter(pl.col.SET==1), "Valid")
        log_print_n(labels.filter(pl.col.SET==0), "Train")
        log_print_n(labels.filter(pl.col.SET==2), "Test")
    else:
        select_fids = np.random.choice(val_labels.filter(pl.col.FINNGENID.is_in(select_fids))["FINNGENID"].to_list(), n_test, replace=False).tolist()
        test_labels = (val_labels
                        .filter(pl.col("FINNGENID").is_in(select_fids))
                        .with_columns(pl.lit(2).cast(pl.Float64).alias("SET"))
                        .unique()
        )
        other_labels = val_labels.filter(~pl.col.FINNGENID.is_in(select_fids))
        other_labels = other_labels.with_columns(pl.lit(1).cast(pl.Float64).alias("SET"))

        logging_print(str(other_labels.height) + " individuals not in selected biobanks for future validation.")

        # All labels are training
        labels = labels.with_columns(pl.lit(0).cast(pl.Float64).alias("SET"))
        val_labels = pl.concat([other_labels, test_labels])
        labels = pl.concat([labels, val_labels])

        log_print_n(labels.filter(pl.col.SET==0), "Train")
        log_print_n(labels.filter(pl.col.SET==1), "Future Valid")
        log_print_n(labels.filter(pl.col.SET==2), "Test")

        val_data = data.filter((pl.col.DATE<val_base_date)&(pl.col.FINNGENID.is_in(val_labels["FINNGENID"])))
        (data
            .select(["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"])
            .write_parquet(args.res_dir + out_file_name + "_val.parquet")
        )
    

    print(labels["SET"].value_counts(normalize=True))
    print(labels["SET"].value_counts(normalize=False))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Data before start                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = data.filter((pl.col.DATE<base_date)&(pl.col.FINNGENID.is_in(labels["FINNGENID"])))

    (data
        .select(["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"])
        .write_parquet(args.res_dir + out_file_name + ".parquet")
    )
    
    labels.write_parquet(args.res_dir + out_file_name + "_labels.parquet")
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
