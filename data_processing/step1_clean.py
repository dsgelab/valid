# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer, get_date, get_datetime, make_dir, init_logging, logging_print, read_file
from processing_utils import custom_abnorm
from clean_utils import remove_severe_value_outliers, remove_single_value_outliers, handle_different_units, handle_missing_values, get_main_unit_data, convert_hba1c_data, remove_known_outliers
# Standard stuff
import pandas as pd
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Info for files
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--file_path", help="Path to extracted file (step0)", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)

    # Processing settings
    parser.add_argument("--fill_missing", type=int, help="Whether to remove missing vlaues or set them to a dummies. [deafult remove: 0]", default=0)
    parser.add_argument("--dummies", type=float, nargs="+", help="List with values for low, normal, high missing value abnormal")
    parser.add_argument("--priority_units", type=str, default=["mmol/mol", "%"], nargs="+", help="List with values for low, normal, high missing value abnormal")
    parser.add_argument("--main_unit", help="Unit that will be filtered out as the main unit.")
    parser.add_argument("--max_z", type=int, help="Maximum z-score among all measurements. [dafult: 10]", default=10)
    parser.add_argument("--ref_min", type=float, help="Minimum reasonable value [dafult: None]", default=None)
    parser.add_argument("--ref_max", type=float, help="Maximum reasonable value [dafult: None]", default=None)
    parser.add_argument("--plot", type=int, help="Whether to create plots for quality control (time-consuming).", default=1)

    args = parser.parse_args()
    return(args)

def replace_missing_with_extracted(data, n_indvs_stats):
    data.group_by(["FINNGENID", "DATE"])
    
    return(data, n_indvs_stats)
if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    # File names and directories
    file_name = args.lab_name + "_d" + str(args.fill_missing) + "_" + get_date() 
    count_dir = args.res_dir + "counts/"
    log_file_name = args.lab_name + "_d" + str(args.fill_missing) + "_" + get_datetime()
    make_dir(args.res_dir); make_dir(count_dir)
    # Logging
    init_logging(args.res_dir, log_file_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Getting data                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #       
    data = read_file(args.file_path)
    # Prep
    n_indvs_stats = pd.DataFrame({"STEP": ["Start", "Dups exact", "Dups manual", "Values", "Outliers_known", "Outliers", "Outliers_single"]})
    n_indv = len(set(data["FINNGENID"]))

    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_INDVS_NOW"] = n_indv

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Processing                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    if args.lab_name != "hba1c":
        data = data.drop(["VALUE", "UNIT"]).rename({"VALUE_FG": "VALUE", "UNIT_FG": "UNIT"})
    if args.lab_name == "hba1c": 
        data = data.drop(["VALUE_FG", "UNIT_FG", "ABNORM_FG"])
        data = data.with_columns(pl.when(pl.col("ABNORM")=="A")
                                   .then(pl.lit("H"))
                                   .otherwise(pl.col("ABNORM"))
                                   .alias("ABNORM"))
    data, n_indvs_stats = handle_different_units(data, 
                                                 n_indvs_stats, 
                                                 logger=logger,
                                                 unit_priority=args.priority_units)
    data, n_indvs_stats = handle_missing_values(data=data, 
                                                n_indvs_stats=n_indvs_stats, 
                                                fill_missing=args.fill_missing, 
                                                dummies=args.dummies, 
                                                dummy_unit=args.main_unit)
    if "ABNORM_FG" in data.columns:
        data = data.drop("ABNORM").rename({"ABNORM_FG": "ABNORM"})
    if args.lab_name != "hba1c":
        data, n_indvs_stats = get_main_unit_data(data=data,
                                                 n_indvs_stats=n_indvs_stats, 
                                                 main_unit=args.main_unit)
    else:
        data, n_indvs_stats = convert_hba1c_data(data=data, 
                                                 n_indvs_stats=n_indvs_stats)

    data, n_indvs_stats = remove_known_outliers(data=data, 
                                                n_indvs_stats=n_indvs_stats, 
                                                ref_min=args.ref_min, 
                                                ref_max=args.ref_max)
    print(n_indvs_stats)
    data, n_indvs_stats = remove_severe_value_outliers(data=data, 
                                                       n_indvs_stats=n_indvs_stats, 
                                                       max_z=args.max_z, 
                                                       res_dir=args.res_dir, 
                                                       file_name=file_name, 
                                                       logger=logger,
                                                       plot=args.plot)
    print(n_indvs_stats)

    if args.lab_name == "egfr" or args.lab_name == "krea":
        data, n_indvs_stats = remove_single_value_outliers(data=data, 
                                                           n_indvs_stats=n_indvs_stats)
    print(n_indvs_stats)

    #### Finishing
    data = custom_abnorm(data, args.lab_name)
    logging.info("Min abnorm " + str(data.filter(pl.col("ABNORM_CUSTOM")>0.0).select(pl.col("VALUE").min()).to_numpy()[0][0]) + " max abnorm " + str(data.filter(pl.col("ABNORM_CUSTOM")>0.0).select(pl.col("VALUE").max()).to_numpy()[0][0]))
    
    # Saving
    data = data.drop(["Z"])
    data.write_parquet(args.res_dir + file_name + ".parquet")
    n_indvs_stats.to_csv(count_dir + file_name + "_counts.csv", sep=",", index=False)
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
