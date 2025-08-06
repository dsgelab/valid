# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer, get_date, get_datetime, make_dir, init_logging, logging_print, read_file
from processing_utils import get_abnorm_func_based_on_name, egfr_ckdepi2021_transform, cystc_ckdepi2012_transform
from clean_utils import remove_severe_value_outliers, remove_single_value_outliers, handle_different_units, handle_missing_values, get_main_unit_data, convert_hba1c_data, remove_known_outliers, handle_same_day_duplicates
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
    parser.add_argument("--keep_last_of_day", help="Keeping last value taken in a day, if at same time will just be random.", default=0)

    parser.add_argument("--max_z", type=int, help="Maximum z-score among all measurements. [dafult: 10]", default=10)
    parser.add_argument("--ref_min", type=float, help="Minimum reasonable value [dafult: None]", default=None)
    parser.add_argument("--ref_max", type=float, help="Maximum reasonable value [dafult: None]", default=None)
    parser.add_argument("--plot", type=int, help="Whether to create plots for quality control (time-consuming).", default=1)
    parser.add_argument("--abnorm_type", type=str, default="", help="[Options: soft, strong]. soft: HbA1c >42 abnormal strong: HbA1c >47 abnormal.")
    parser.add_argument("--transform_egfr", type=int, default=1, help="Whether to transform krea or cystatin C to eGFR.")
    parser.add_argument("--min_age", type=int, default=18, help="Minimum age to filter for, this will make outlier removal easier.")

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
    file_name = args.lab_name + "_d" + str(args.fill_missing)
    # File names and directories
    if args.abnorm_type != "":
         file_name = file_name + "_" + args.abnorm_type
    if args.keep_last_of_day:
        file_name = file_name + "_ld"
    file_name = file_name + "_" + get_date() 

    count_dir = args.res_dir + "counts/"
    make_dir(args.res_dir); make_dir(count_dir)
    # Logging
    init_logging(args.res_dir, args.lab_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Getting data                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #       
    data = read_file(args.file_path)
    data = data.with_columns(pl.col.DATE.dt.datetime().alias("DATE"))
    r13_indivs = pl.read_csv("/finngen/library-red/all_allowed_ids_to_sb/finngen_R13_finngenid_actual_inclusion_list.txt", has_header=False)
    print(f"Removing individuals not in R13: {data.filter(~pl.col.FINNGENID.is_in(r13_indivs["column_1"]))["FINNGENID"].unique().len()}")
    data = data.filter(pl.col.FINNGENID.is_in(r13_indivs["column_1"]))
    
    # Prep
    n_indvs_stats = pd.DataFrame({"STEP": ["Start", "Dups manual", "Values", "Dups mean", "Outliers_known", "Outliers", "Outliers_single"]})
    n_indv = len(set(data["FINNGENID"]))

    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_ROWS_NOW"] = data.height
    n_indvs_stats.loc[n_indvs_stats.STEP == "Start","N_INDVS_NOW"] = n_indv

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Processing                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    data = data.filter(pl.col.EVENT_AGE>=args.min_age)
    if args.lab_name == "hba1c" or args.lab_name == "tsh": 
        data = data.drop(["VALUE_FG", "UNIT_FG", "ABNORM_FG"])
    else:
        data = data.drop(["VALUE", "UNIT"]).rename({"VALUE_FG": "VALUE", "UNIT_FG": "UNIT"})
    if args.lab_name == "uacr":
        # there is something very weird about the low values
        data = data.filter(pl.col.ABNORM!="L")
    if args.lab_name == "tsh":
        # It really seems they are not to be multiplied same distribution and all
        data = (data.with_columns(pl.when((pl.col.UNIT=="u/l")|(pl.col.UNIT=="iu/l"))
                                  .then(pl.lit("mu/l"))
                                  .otherwise(pl.col.UNIT)
                                  .alias("UNIT")))
    if args.lab_name == "hba1c" or args.lab_name == "tsh":
        data = data.with_columns(pl.when(pl.col("ABNORM")=="A")
                                   .then(pl.lit("H"))
                                   .otherwise(pl.col("ABNORM"))
                                   .alias("ABNORM"))
    if args.lab_name != "ogtt":
        data, n_indvs_stats = handle_different_units(data, 
                                                     n_indvs_stats, 
                                                     unit_priority=args.priority_units)
        data, n_indvs_stats = handle_missing_values(data=data, 
                                                    n_indvs_stats=n_indvs_stats, 
                                                    fill_missing=args.fill_missing, 
                                                    dummies=args.dummies, 
                                                    dummy_unit=args.main_unit)
    else:
        data = data.filter(~pl.col.ABNORM.is_null())
    print(data["VALUE"].describe())
    if "ABNORM_FG" in data.columns:
        data = (data
                .drop("ABNORM")
                .with_columns(
                    pl.when(pl.col.ABNORM_FG=="N").then(0)
                    .when(pl.col.ABNORM_FG=="L").then(-1)
                    .when(pl.col.ABNORM_FG=="H").then(1)
                    .when(pl.col.ABNORM_FG=="LL").then(-2)
                    .when(pl.col.ABNORM_FG=="HH").then(2)
                    .cast(pl.Float64).alias("ABNORM")
                )
               )
    print(data["VALUE"].describe())

    if args.lab_name != "ogtt":
        if args.lab_name != "hba1c":
            data, n_indvs_stats = get_main_unit_data(data=data,
                                                     n_indvs_stats=n_indvs_stats, 
                                                     main_unit=args.main_unit)
        else:
            data, n_indvs_stats = convert_hba1c_data(data=data, 
                                                     n_indvs_stats=n_indvs_stats)
        data, n_indvs_stats = handle_same_day_duplicates(data, n_indvs_stats, args.keep_last_of_day)
        if args.lab_name == "egfr":
            data = egfr_ckdepi2021_transform(data, "VALUE")
        if args.lab_name == "cystc" and args.transform_egfr == 1:
            data = cystc_ckdepi2012_transform(data, "VALUE")
        data, n_indvs_stats = remove_known_outliers(data=data, 
                                                    n_indvs_stats=n_indvs_stats, 
                                                    ref_min=args.ref_min, 
                                                    ref_max=args.ref_max)
        data, n_indvs_stats = remove_severe_value_outliers(data=data, 
                                                               n_indvs_stats=n_indvs_stats, 
                                                               max_z=args.max_z, 
                                                               res_dir=args.res_dir, 
                                                               file_name=file_name, 
                                                               logger=logger,
                                                               plot=args.plot)
        print(data["VALUE"].describe())

    if args.lab_name == "egfr":
        data, n_indvs_stats = remove_single_value_outliers(data=data, 
                                                           n_indvs_stats=n_indvs_stats)
    print(data["VALUE"].describe())


    #### Finishing
    try:
        if args.abnorm_type != "": 
            data =  get_abnorm_func_based_on_name(args.lab_name, args.abnorm_type)(data, "VALUE")
        else: 
            data =  get_abnorm_func_based_on_name(args.lab_name)(data, "VALUE")
    except ValueError:
        print("couldnt do custom abnormality, using FinnGen processed column.")
        data = data.rename({"ABNORM": "ABNORM_CUSTOM"})
    logging.info("Min abnorm " + str(data.filter(pl.col("ABNORM_CUSTOM")>0.0).select(pl.col("VALUE").min()).to_numpy()[0][0]) + " max abnorm " + str(data.filter(pl.col("ABNORM_CUSTOM")>0.0).select(pl.col("VALUE").max()).to_numpy()[0][0]))
    print(data)
    # Saving
    if "Z" in data.columns:
        data = data.drop(["Z"])
    data.write_parquet(args.res_dir + file_name + ".parquet")
    n_indvs_stats.to_csv(count_dir + file_name + "_counts.csv", sep=",", index=False)
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
