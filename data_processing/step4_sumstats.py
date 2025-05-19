

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



def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--file_path", type=str, help="Path to the data file", required=True)
    parser.add_argument("--file_name_start", type=str, help="Name of the data file", required=True)
    parser.add_argument("--file_path_labels", type=str, help="Path to the diagnosis data file", default="")
    parser.add_argument("--file_path_data", type=str, help="Path to the diagnosis data file", default="")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--start_date", type=str, default="", help="Date to filter before")


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
    init_logging(args.res_dir, args.lab_name, logger, args)
    out_file_name = args.file_name_start+"_sumstats_"+get_date()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    
    
    if args.file_path_labels != "":
        data = read_file(args.file_path_data)
        labels = read_file(args.file_path_labels)
    else:
        data = read_file(args.file_path+args.file_name_start+".parquet")
        labels = read_file(args.file_path+args.file_name_start+"_labels.parquet")
    data = data.join(labels, on="FINNGENID", how="right", coalesce=True)
    if args.start_date != "": 
        data = data.with_columns(pl.Series("START_DATE", [datetime.strptime(args.start_date, "%Y-%m-%d")]*data.height))
    data = data.filter(pl.col.DATE<pl.col.START_DATE)
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Summary statistics                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    sumstats = (data
                .filter((pl.col.VALUE.is_not_null()))
                .sort(["FINNGENID", "DATE"], descending=False)
                .group_by(["FINNGENID", "SET"])
                .agg(
                pl.col.VALUE.mean().alias("MEAN"),
                pl.col.VALUE.std().alias("SD"),
                pl.col.VALUE.min().alias("MIN"),
                pl.col.VALUE.max().alias("MAX"),
                pl.col.VALUE.sum().alias("SUM"),
                pl.col.VALUE.kurtosis().alias("KURT"),
                pl.col.VALUE.skew().alias("SKEW"),
                (pl.col.VALUE**2).sum().alias("ABS_ENERG"),
                (pl.col.VALUE-pl.col.VALUE.shift(1)).sum().alias("SUM_CHANGE"),
                (pl.col.VALUE-pl.col.VALUE.shift(1)).mean().alias("MEAN_CHANGE"),
                (pl.col.VALUE.count()).cast(pl.Int64).alias("SEQ_LEN"),
                (pl.col.VALUE.quantile(0.25)).alias("QUANT_25"),
                (pl.col.VALUE.quantile(0.75)).alias("QUANT_75"),
                (pl.col.VALUE.get(pl.col.DATE.arg_min())).alias("IDX_QUANT_0"),
                (pl.col.VALUE.get(pl.col.DATE.arg_max())).alias("IDX_QUANT_100"),
                (pl.col.ABNORM_CUSTOM.sum()).cast(pl.Int64).alias("ABNORM"),
                # Distance of minimum compared to prediction in days
                (((pl.col.DATE.get(pl.col.VALUE.arg_min())-pl.col.START_DATE.first()).dt.total_days()/365.25).abs()).alias("MIN_LOC"),
                (((pl.col.DATE.get(pl.col.VALUE.arg_max())-pl.col.START_DATE.first()).dt.total_days()/365.25).abs()).alias("MAX_LOC"),
                ((pl.col.DATE.max()-pl.col.DATE.min()).dt.total_days()/365.25).alias("FIRST_LAST"),
                (pl.col.DATE.max()).alias("LAST_VAL_DATE")
                )
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Mean for those with missing data                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    sumstats_train = sumstats.filter(pl.col.SET==0)
    if args.file_path_labels=="":
        # atm this only works for the original data as labels then has empty rows. Otherwise we just dont have data.
        missing_data = data.filter((pl.col.VALUE.is_null())&((pl.len()==1).over("FINNGENID"))).select("FINNGENID", "SET")
    
        missing_data = (missing_data.with_columns(
                            pl.Series("MIN", [sumstats_train["MIN"].mean()]*missing_data.height),
                            pl.Series("MAX", [sumstats_train["MAX"].mean()]*missing_data.height),
                            pl.Series("MEAN", [sumstats_train["MEAN"].mean()]*missing_data.height),
                            pl.Series("SUM", [sumstats_train["SUM"].mean()]*missing_data.height),
                            pl.Series("ABS_ENERG", [sumstats_train["ABS_ENERG"].mean()]*missing_data.height),
                            pl.Series("QUANT_25", [sumstats_train["QUANT_25"].mean()]*missing_data.height),
                            pl.Series("QUANT_75", [sumstats_train["QUANT_75"].mean()]*missing_data.height),
                            pl.Series("IDX_QUANT_0", [sumstats_train["IDX_QUANT_0"].mean()]*missing_data.height),
                            pl.Series("IDX_QUANT_100", [sumstats_train["IDX_QUANT_100"].mean()]*missing_data.height),
                            pl.Series("MIN_LOC", [sumstats_train["MIN_LOC"].mean()]*missing_data.height).cast(pl.Float64),
                            pl.Series("MAX_LOC", [sumstats_train["MAX_LOC"].mean()]*missing_data.height).cast(pl.Float64),
                            pl.Series("KURT", [None]*missing_data.height).cast(pl.Float64, strict=False),
                            pl.Series("SKEW", [None]*missing_data.height).cast(pl.Float64, strict=False),
                            pl.Series("SD", [None]*missing_data.height).cast(pl.Float64, strict=False),
                            pl.Series("MEAN_CHANGE", [None]*missing_data.height).cast(pl.Float64, strict=False),
                            pl.Series("SUM_CHANGE", [0]*missing_data.height).cast(pl.Float64, strict=False),
                            pl.Series("LAST_VAL_DATE", [None]*missing_data.height).cast(pl.Date, strict=False),
                            pl.Series("SEQ_LEN", [0]*missing_data.height).cast(pl.Int64, strict=False),
                            pl.Series("FIRST_LAST", [0]*missing_data.height).cast(pl.Float64, strict=False),
                            pl.Series("ABNORM", [0]*missing_data.height).cast(pl.Int64, strict=False),
                        )
        )
        sumstats = pl.concat([sumstats, missing_data.select(sumstats.columns)])
    print(sumstats)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    sumstats.write_parquet(args.res_dir+out_file_name+".parquet")