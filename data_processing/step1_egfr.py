# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
# Standard stuff
import pandas as pd
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse

# Statistics/Processing
import scipy.stats as st
from general_utils import get_date, make_dir, init_logging, Timer
from clean_utils import remove_severe_value_outliers, remove_single_value_outliers, handle_exact_duplicates, handle_same_day_duplicates
from processing_utils import get_abnorm_func_based_on_name

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Path to extracted file (step0)", required=True)
    parser.add_argument("--res_dir", help="Path to the results directory (step1)", default="/home/ivm/valid/data/processed_data/step1/")
    parser.add_argument("--lab_name", help="Readable name of the measurement value.", required=True)
    parser.add_argument("--max_z", type=int, help="Maximum z-score among all measurements. [dafult: 10]", default=10)
    parser.add_argument("--plot", type=int, help="Minimum reasonable value [dafult: None]", default=1)
    parser.add_argument("--abnorm_type", type=str, default="KDIGO-strict", help="[Options: age, KDIGO-strict, KDIGO-soft]. age: egfr abnormality based on age. KDIGO-stric: <60 for all. KDIGO-soft: <60 but with 60-65 allowed in between abnormal without disrupting the count.")

    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    
    file_name = args.lab_name + "_" + args.abnorm_type + "_" + get_date() 
    count_dir = args.res_dir + "counts/"
    make_dir(args.res_dir); make_dir(count_dir)
    
    init_logging(args.res_dir, args.lab_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Getting data                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
    data = pl.DataFrame(pd.read_csv(args.file_path, sep="\t", compression="gzip"))
    # Prep
    n_indvs_stats = pd.DataFrame({"STEP": ["Start", "Dups exact", "Dups mean", "Outliers", "Outliers_single"]})
    n_indv = len(set(data["FINNGENID"]))

    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_ROWS_NOW"] = data.shape[0]
    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_INDVS_NOW"] = n_indv

    data = data.select(["FINNGENID", "SEX", "EVENT_AGE", "APPROX_EVENT_DATETIME", "egfr_ckdepi2021"])
    data = data.rename({"APPROX_EVENT_DATETIME": "DATE", "egfr_ckdepi2021": "VALUE"})

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Processing                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    data, n_indvs_stats = handle_exact_duplicates(data=data, 
                                                  n_indvs_stats=n_indvs_stats)
    data, n_indvs_stats = handle_same_day_duplicates(data=data,
                                                     n_indvs_stats=n_indvs_stats)
    data, n_indvs_stats = remove_severe_value_outliers(data=data,
                                                       n_indvs_stats=n_indvs_stats, 
                                                       max_z=args.max_z, 
                                                       res_dir=args.res_dir, 
                                                       file_name=file_name, 
                                                       logger=logger,
                                                       plot=args.plot)
    data, n_indvs_stats = remove_single_value_outliers(data=data, 
                                                       n_indvs_stats=n_indvs_stats)
    #### Finishing
    data = get_abnorm_func_based_on_name(args.lab_name, args.abnorm_type)(data, "VALUE")
    logging.info("Min abnorm " + str(data.filter(pl.col("ABNORM_CUSTOM")>0.0).select(pl.col("VALUE").min()).to_numpy()[0][0]) + " max abnorm " + str(data.filter(pl.col("ABNORM_CUSTOM")>0.0).select(pl.col("VALUE").max()).to_numpy()[0][0]))
    
    # Saving
    data = data.drop(["Z"])
    data.write_parquet(args.res_dir + file_name + ".parquet")
    n_indvs_stats.to_csv(count_dir + file_name + "_counts.csv", sep=",", index=False)
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
