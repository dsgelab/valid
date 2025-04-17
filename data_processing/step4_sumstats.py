

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


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--file_path", type=str, help="Path to the data file", required=True)
    parser.add_argument("--file_name_start", type=str, help="Name of the data file", required=True)
    parser.add_argument("--file_path_labels", type=str, help="Path to the diagnosis data file", default="")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)

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
    data = read_file(args.file_path+args.file_name_start+".parquet")
    if args.file_path_labels != "":
        labels = read_file(args.file_path_labels)
    else:
        labels = read_file(args.file_path+args.file_name_start+"_labels.parquet")
    data = data.join(labels, on="FINNGENID", how="right", coalesce=True)
    print(data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Summary statistics                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    sumstats = (data
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
                (pl.col.VALUE-pl.col.VALUE.shift(1)).abs().sum().alias("SUM_ABS_CHANGE"),
                (pl.col.VALUE-pl.col.VALUE.shift(1)).abs().mean().alias("MEAN_ABS_CHANGE"),
                (pl.col.VALUE-pl.col.VALUE.shift(1)).abs().max().alias("MAX_ABS_CHANGE"),
                (pl.col.VALUE-pl.col.VALUE.shift(1)).max().alias("MAX_CHANGE"),
                (pl.col.VALUE-pl.col.VALUE.shift(1)).min().alias("MIN_CHANGE"),
                (pl.col.VALUE-pl.col.VALUE.shift(1)).mean().alias("MEAN_CHANGE"),
                (pl.col.VALUE.count()).alias("SEQ_LEN"),
                (pl.col.VALUE.quantile(0.25)).alias("QUANT_25"),
                (pl.col.VALUE.quantile(0.5)).alias("QUANT_75"),
                (pl.col.VALUE.get(pl.col.DATE.arg_min())).alias("IDX_QUANT_0"),
                (pl.col.VALUE.get(pl.col.DATE.arg_max())).alias("IDX_QUANT_100"),
                (pl.col.ABNORM_CUSTOM.sum()).alias("ABNORM"),
                # Distance of minimum compared to prediction in days
                (pl.col.DATE.get(pl.col.VALUE.arg_min())-pl.col.START_DATE).dt.days().alias("MIN_LOC"),
                (pl.col.DATE.get(pl.col.VALUE.arg_max())-pl.col.START_DATE).dt.days().alias("MAX_LOC"),
                (pl.col.DATE.max()-pl.col.DATE.min()).dt.total_days().alias("FIRST_LAST"),
                (pl.col.DATE.max()).dt.to_date().alias("LAST_VAL_DATE")
                )
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Mean for those with missing data                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    sumstats_train = sumstats.filter(pl.col.SET==0)
    missing_data = data.filter((pl.col.VALUE.is_null())&((pl.len()==1).over("FINNGENID")))
    missing_data = (missing_data.with_columns(
                        sumstats_train["MIN"].mean().alias("MIN"),
                        sumstats_train["MAX"].mean().alias("MAX"),
                        sumstats_train.select(pl.col("MEAN")).mean().alias("MEAN"),
                        sumstats_train.select(pl.col("SUM")).mean().alias("SUM"),
                        sumstats_train.select(pl.col("ABS_ENERG")).mean().alias("ABS_ENERG"),
                        sumstats_train.select(pl.col("QUANT_25")).mean().alias("QUANT_25"),
                        sumstats_train.select(pl.col("QUANT_75")).mean().alias("QUANT_75"),
                        sumstats_train.select(pl.col("IDX_QUANT_0")).mean().alias("IDX_QUANT_0"),
                        sumstats_train.select(pl.col("IDX_QUANT_100")).mean().alias("IDX_QUANT_100"),
                        sumstats_train.select(pl.col("MIN_LOC")).mean().alias("MIN_LOC"),
                        sumstats_train.select(pl.col("MAX_LOC")).mean().alias("MAX_LOC"),
                        pl.Null.alias("KURT"),
                        pl.Null.alias("SKEW"),
                        pl.Null.alias("SD"),
                        pl.Null.alias("SUM_ABS_CHANGE"),
                        pl.Null.alias("MEAN_ABS_CHANGE"),
                        pl.Null.alias("MAX_CHANGE"),
                        pl.Null.alias("MIN_CHANGE"),
                        pl.Null.alias("MAX_ABS_CHANGE"),
                        pl.Null.alias("MEAN_CHANGE"),
                        pl.Null.alias("LAST_VAL_DATE"),
                        pl.Lit(0).alias("SEQ_LEN"),
                        pl.Lit(0).alias("FIRST_LAST"),
                        pl.Lit(0).alias("ABNORM"),
                    )
    )
    sumstats = pl.concat([sumstats, missing_data.select(sumstats.columns)])

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    sumstats.write_parquet(args.res_dir+out_file_name+".parquet")