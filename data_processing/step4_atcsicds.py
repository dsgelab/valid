# Custom utils
import sys

sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, init_logging, Timer, read_file

# Standard stuff
import polars as pl
import argparse
# Logging
import logging
logger = logging.getLogger(__name__)
from datetime import datetime

def get_out_file_name(file_name_start: str,
                      lab_name: str,
                      col_name: str,
                      time: int,
                      bin_count: int,
                      months_before: int,
                      start_year: int,
                     min_pct: str) -> str:
    """
    Function to get the output file name based on the input parameters.
    """

    out_file_name = file_name_start
    if col_name == "ICD_THREE":
        out_file_name += "_icds"
    elif col_name == "ATC_FIVE":
        out_file_name += "_atcs"
    out_file_name += "_"+min_pct+"pct"
    if bin_count == 1:
        out_file_name += "_bin"
    else:
        out_file_name += "_count"
    if time == 1:
        out_file_name += "_current"
    elif time == -1:
        out_file_name += "_historical"
    if months_before > 0:
        out_file_name += "_buffer" + str(months_before) + "months"
    if start_year > 0:
        out_file_name += "_start_" + str(start_year)
    out_file_name += "_" + lab_name + "_" + get_date()
    
    return out_file_name

def get_time_period_data(file_path_long: str,
                         preds_data: pl.DataFrame, 
                         time: int, 
                         months_before: int) -> pl.DataFrame:
    """
    Function to get the time period data based on the input parameters.
        historical - all predictor information only collected before first measurement -X months
        current - all predictor information only collected after first measurement -X months
    """
    # Need date of first measurement of each individual to filter historical or current data
    start_dates = read_file(file_path_long)
    start_dates = (start_dates
        .sort(["FINNGENID", "DATE"])
        .agg(pl.col.DATE==pl.col.DATE.first().over("FINNGENID").alias("DATA_START_DATE"))  # Date of first recorded measurement for each individual
    )
    # Start date of information X-months before first measurement
    start_dates = start_dates.with_columns(
        (pl.col.DATA_START_DATE - pl.duration(months=months_before)).alias("DATA_START_DATE")
    )
    # Adding info to predictor data
    preds_data = preds_data.join(start_dates, on="FINNGENID", how="left")
    # Filtering out only current or historical data
    if time == 1:
        preds_data = preds_data.filter(pl.col("DATE") >= pl.col("DATA_START_DATE")).drop("DATA_START_DATE")
    elif time == -1:
        preds_data = preds_data.filter(pl.col("DATE") <= pl.col("DATA_START_DATE")).drop("DATA_START_DATE")
    return preds_data

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--file_path_preds", type=str, help="Full path to the longitduinal ICD or ATC data.", required=True)
    parser.add_argument("--dir_path_labels", type=str, help="Path to directory with label data.", required=True)
    parser.add_argument("--file_name_labels_start", type=str, help="Path to the diagnosis data file", default="")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)

    # Settings
    parser.add_argument("--min_pct", type=str, help="Min percentage of inclusion for file name.", required=True)
    parser.add_argument("--col_name", type=str, help="Name of the column to use from the longitduinal data. [i.e. ATC_FIVE or ICD_THREE]", required=True)
    parser.add_argument("--time", type=int, default=0, help="Whether to filter for current and not historical data. -1 = historical, 0 = all, 1 = current")
    parser.add_argument("--bin_count", type=int, default=1, help="Whether to count number of occurance or stay binary observed/not observed.")
    parser.add_argument("--months_before", type=int, default=0, help="Months to add before start of measurements for a buffer.")
    parser.add_argument("--start_year", type=int, default=0, help="What year to start the data from. Ignored if 0.")
    parser.add_argument("--start_date", type=str, default="", help="Date to filter before", required=False)

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
    out_file_name = get_out_file_name(args.file_name_labels_start, 
                                      args.lab_name, 
                                      args.col_name, 
                                      args.time, 
                                      args.bin_count, 
                                      args.months_before, 
                                      args.start_year,
                                     args.min_pct)
    init_logging(args.res_dir, args.lab_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    preds_data = read_file(args.file_path_preds)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Filter to right time window                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    if args.time != 0:
        preds_data = get_time_period_data(args.dir_path_labels+args.file_name_labels_start+".parquet",
                                          preds_data,
                                          args.time,
                                          args.months_before)
    labels = read_file(args.dir_path_labels+args.file_name_labels_start+"_labels.parquet")
    if args.start_date == "": 
        # Filtering out data before start of prediction   
        preds_data = preds_data.join(labels.select("FINNGENID", "START_DATE"), on="FINNGENID", how="left")
        preds_data = preds_data.filter(pl.col.DATE < pl.col.START_DATE).drop("START_DATE")
    else:
        preds_data = preds_data.filter(pl.col.DATE < datetime.strptime(args.start_date, "%Y-%m-%d"))
    # Filtering out data only after what we set as start year
    if args.start_year != 0:
        preds_data = preds_data.filter(pl.col.DATE.dt.year() >= args.start_year)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Counting and wide                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    preds_wider = (preds_data
                    .group_by(["FINNGENID", args.col_name])
                    .agg(pl.len().alias("N_CODE"))
    )

    # Making binary 0/1 observed if bin_count is 1
    if args.bin_count == 1:
        preds_wider = preds_wider.with_columns(pl.when(pl.col.N_CODE>=1).then(1).otherwise(0).alias("N_CODE"))

    # Making wide
    preds_wider = (preds_wider
                    .pivot(values="N_CODE", index="FINNGENID", on=args.col_name)
                    .fill_null(0)
    )

    # Add people without any codes
    preds_wider = preds_wider.join(labels[["FINNGENID"]], on="FINNGENID", how="full", coalesce=True)
    preds_wider = preds_wider.fill_null(0)
    print(preds_wider)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Date of last code                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    last_date = (preds_data
                    .select(["FINNGENID", "DATE"])
                    .unique()
                    .sort(["FINNGENID", "DATE"], descending=True)
                    .group_by("FINNGENID")
                    .agg(pl.col("DATE").first().alias("LAST_CODE_DATE"))
    )

    # Adding info to predictor data
    preds_wider = preds_wider.join(last_date, on="FINNGENID", how="left")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    preds_wider.write_parquet(args.res_dir+out_file_name+".parquet")