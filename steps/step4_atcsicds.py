# Custom utils
import sys

sys.path.append(("../utils/"))
from general_utils import get_date, init_logging, Timer, read_file, logging_print
from input_utils import get_all_indvs

# Standard stuff
import polars as pl
import pandas as pd
import argparse
# Logging
import logging
logger = logging.getLogger(__name__)
from datetime import datetime

def get_out_file_name(col_name: str,
                      bin_count: int,
                      start_year: int,
                      min_pct: str,
                      start_date: str) -> str:
    """
    Function to get the output file name based on the input parameters.
    """

    if col_name == "ICD_THREE":
        out_file_name = "icds"
    elif col_name == "ATC_FIVE":
        out_file_name = "atcs"
    out_file_name += "_"+str(int(min_pct))+"pct"
    if bin_count == 1:
        out_file_name += "_bin"
    else:
        out_file_name += "_count"
    if start_year > 0:
        out_file_name += "_start_" + str(start_year)
    if start_date != "": 
        out_file_name = out_file_name + "_pred" + str(int(start_date[0:4])+1) + "_"
        
    out_file_name += get_date()
    
    return out_file_name

def prep_preds_data(preds_data: pl.DataFrame,
                    start_date: str,
                    start_year: int) -> pl.DataFrame:
    """
    Function to prepare the predictor data based on the input parameters.
    """
    # If date column string
    if preds_data["DATE"].dtype == pl.Utf8:
        preds_data = preds_data.with_columns(pl.col("DATE").str.to_date().alias("DATE"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Filter to right time window                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    try:
        preds_data = preds_data.filter(pl.col.DATE < datetime.strptime(start_date, "%Y-%m-%d"))
    except:
        preds_data = preds_data.filter(pl.col.DATE < datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S"))

    # Filtering out data only after what we set as start year
    if start_year != 0:
        preds_data = preds_data.filter(pl.col.DATE.dt.year() >= start_year)

    return preds_data

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--file_path_preds", type=str, help="Full path to the longitduinal ICD or ATC data.", required=True)

    # Settings
    parser.add_argument("--min_pct", type=int, help="Min percentage of inclusion for file name. Percentage 0-100.", required=True)
    parser.add_argument("--col_name", type=str, help="Name of the column to use from the longitduinal data. [i.e. ATC_FIVE or ICD_THREE]", required=True)
    parser.add_argument("--bin_count", type=int, default=1, help="Whether to count number of occurance or stay binary observed/not observed.")
    parser.add_argument("--start_year", type=int, default=0, help="What year to start the data from. Ignored if 0.")
    parser.add_argument("--start_date", type=str, default="", help="Date to filter before.", required=False)
    parser.add_argument("--fg_ver", type=str, default="R14", help="Version/SB. Can be R12-14 or ml4h.", required=False)

    parser.add_argument("--select_code_paths", type=str, default="", help="List of paths to selected code files.", required=False)

    # Settings
    args = parser.parse_args()
    return(args)

#do the main file functions and runs 
if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    out_file_name = get_out_file_name(args.col_name, 
                                      args.bin_count, 
                                      args.start_year,
                                      args.min_pct,
                                      args.start_date)
    init_logging(args.res_dir, "all", logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Counting and wide in one go                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.select_code_paths == "":
        preds_data = read_file(args.file_path_preds)
        preds_data = prep_preds_data(preds_data=preds_data,
                                     start_date=args.start_date,
                                     start_year=args.start_year) 
        preds_wider = (preds_data
                        .group_by(["FINNGENID", args.col_name])
                        .agg(pl.len().alias("N_CODE"))
        )

        # Making binary 0/1 observed if bin_count is 1
        if args.bin_count == 1:
            preds_wider = preds_wider.with_columns(pl.when(pl.col.N_CODE>=1).then(1).otherwise(0).alias("N_CODE"))

        preds_wider = (preds_wider
                        .pivot(values="N_CODE", index="FINNGENID", on=args.col_name)
        )
    else:
        # Making sure to only select the codes that are in at least x% of individuals.
        select_icds = (read_file(args.select_code_paths)
                       .filter(pl.col.PCT_INDV >= float(args.min_pct/100))
                       .sort(args.col_name)
                       .get_column(args.col_name)
                       )
        
        preds_wider = pl.DataFrame()
        total_timer = Timer()
        for code in select_icds:
            code_timer = Timer()

            code_data = pl.DataFrame(pd.read_parquet(args.file_path_preds, filters=[(f"{args.col_name}", "==", code)]))
            
            if "FINNGENID" not in code_data.columns:
                code_data = code_data.rename({"FID": "FINNGENID"})
            code_data = prep_preds_data(preds_data=code_data,
                                        start_date=args.start_date,
                                        start_year=args.start_year)
            code_data = code_data.group_by("FINNGENID").agg(pl.len().alias("N_CODE"))

            if args.bin_count == 1:
                code_data = code_data.with_columns(pl.when(pl.col.N_CODE>=1).then(1).otherwise(0).alias("N_CODE"))
            code_data = code_data.rename({"N_CODE": code})
            if preds_wider.is_empty():
                preds_wider = code_data
            else:
                preds_wider = preds_wider.join(code_data, on="FINNGENID", how="full", coalesce=True)
            logging_print("Time for code: " + code + " took " + str(code_timer.get_elapsed()))
        preds_wider = preds_wider.fill_null(0)
        logging_print(f"Total time for looping through omops: {total_timer.get_elapsed()}")

    # Add people without any codes
    all_indvs = get_all_indvs(args.fg_ver)

    preds_wider = preds_wider.join(all_indvs[["FINNGENID"]], on="FINNGENID", how="full", coalesce=True)
    preds_wider = preds_wider.fill_null(0)
    print(preds_wider)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    logging_print(args.res_dir+out_file_name+".parquet")
    preds_wider.write_parquet(args.res_dir+out_file_name+".parquet")