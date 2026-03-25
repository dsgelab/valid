

# Custom utils
import sys

sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, init_logging, Timer, read_file
from processing_utils import egfr_ckdepi2021_transform

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
    parser.add_argument("--file_path_data_2", type=str, help="Path to the diagnosis data file", default="")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--start_date", type=str, default="", help="Date to filter before")
    parser.add_argument("--mean_impute", type=int, default=1, help="Whether to impute mean for those with missing. [Options: 0 (No), 1 (Yes some), 2 (Yes all)]")
    parser.add_argument("--interpolate", type=int, default=0, help="Whether to first mean aggregate the data for each month and then interpolate the missing months.")

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
    if args.mean_impute >= 1 and not args.interpolate >= 1:
        out_file_name = args.file_name_start+"_sumstats_"
    elif args.mean_impute >= 1 and args.interpolate >= 1:
        out_file_name = args.file_name_start+"_sumstats_int_"
    elif args.mean_impute == 0 and args.interpolate >= 1:
        out_file_name = args.file_name_start+"_sumstats_ni_int_"
    else:
        out_file_name = args.file_name_start+"_sumstats_ni_"
    if args.start_date != "": 
        out_file_name = out_file_name + "pred" + str(int(args.start_date[0:4])+1) + "_"
    out_file_name = out_file_name + get_date()
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    
    
    if args.file_path_labels != "":
        print(args.file_path_data)
        data = read_file(args.file_path_data)
        if args.file_path_data_2 != "":
            data_2 = read_file(args.file_path_data_2)
            data = pl.concat([data, data_2])
        labels = read_file(args.file_path_labels)
    else:
        data = read_file(args.file_path+args.file_name_start+".parquet")
        labels = read_file(args.file_path+args.file_name_start+"_labels.parquet")
    data = data.join(labels, on="FINNGENID", how="right", coalesce=True)
    if args.start_date != "": 
        try:
            data = data.with_columns(pl.Series("START_DATE", [datetime.strptime(args.start_date, "%Y-%m-%d")]*data.height))
        except:
            data = data.with_columns(pl.Series("START_DATE", [datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S")]*data.height))

    if args.interpolate == 1:
        print(data.columns)
        no_data = data.filter(pl.col.VALUE.is_null()).select(["FINNGENID", "SEX", "EVENT_AGE", "VALUE", "DATE", "START_DATE", "SET", "ABNORM_CUSTOM"])
        data = data.filter(pl.col.VALUE.is_not_null())
        month_data = (data
                      .filter(pl.col.DATE<pl.col.START_DATE)
                      .with_columns(pl.col.DATE.dt.truncate("1mo").alias("DATE"))
                      .group_by("FINNGENID", "DATE")
                      .agg(pl.col.VALUE.mean(),
                           pl.col.SEX.first(),
                           pl.col.START_DATE.first(),
                           pl.col.SET.first(),
                           pl.col.EVENT_AGE.min()+((pl.col.EVENT_AGE.max()-pl.col.EVENT_AGE.min())/2))
        )
        all_months = (month_data
                         .group_by("FINNGENID")
                         .agg(pl.date_range(pl.col.DATE.min(), pl.col.DATE.max(), interval="1mo").alias("DATE"))
                         .explode("DATE")
        )
        data = (month_data
                    .join(all_months, on=["FINNGENID", "DATE"], how="right")
                    .sort(["FINNGENID", "DATE"])
                    .with_columns(pl.col.VALUE.interpolate().over("FINNGENID").alias("VALUE"),
                                  pl.col.EVENT_AGE.interpolate().over("FINNGENID").alias("EVENT_AGE"),
                                  pl.col.SEX.first().over("FINNGENID").alias("SEX")
                                 )
                    
        ).select(["FINNGENID", "SEX", "EVENT_AGE", "VALUE", "DATE", "START_DATE", "SET"])
        print(data)
        data = egfr_ckdepi2021_transform(data, "VALUE")
        data = data.with_columns(pl.when(pl.col.VALUE<60).then(pl.lit(1)).otherwise(pl.lit(0)).alias("ABNORM_CUSTOM").cast(pl.Float64))
        data = pl.concat([no_data, data])
        print(data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Summary statistics                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    sumstats = (data
                .filter(pl.col.DATE<pl.col.START_DATE)
                .filter((pl.col.VALUE.is_not_null()))
                .sort(["FINNGENID", "DATE"], descending=False)
                .group_by(["FINNGENID", "SET"])
                .agg(
                pl.col.VALUE.mean().alias("MEAN"),
                pl.lit(0).alias("NO_HISTORY"),
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
    if args.file_path_labels=="" and args.mean_impute >= 1:
        # atm this only works for the original data as labels then has empty rows. Otherwise we just dont have data.
        missing_data = data.filter((pl.col.VALUE.is_null())&((pl.len()==1).over("FINNGENID"))).select("FINNGENID", "SET")

        if args.mean_impute == 1:
            missing_data = (missing_data.with_columns(
                                pl.Series("MIN", [sumstats_train["MIN"].mean()]*missing_data.height),
                                pl.Series("MAX", [sumstats_train["MAX"].mean()]*missing_data.height),
                                pl.Series("MEAN", [sumstats_train["MEAN"].mean()]*missing_data.height),
                                pl.Series("NO_HISTORY", [1]*missing_data.height).cast(pl.Int32, strict=False),
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
        elif args.mean_impute == 2:
           missing_data = (missing_data.with_columns(
                                pl.Series("MIN", [sumstats_train["MIN"].mean()]*missing_data.height),
                                pl.Series("MAX", [sumstats_train["MAX"].mean()]*missing_data.height),
                                pl.Series("MEAN", [sumstats_train["MEAN"].mean()]*missing_data.height),
                                pl.Series("NO_HISTORY", [1]*missing_data.height).cast(pl.Int32, strict=False),
                                pl.Series("SUM", [sumstats_train["SUM"].mean()]*missing_data.height),
                                pl.Series("ABS_ENERG", [sumstats_train["ABS_ENERG"].mean()]*missing_data.height),
                                pl.Series("QUANT_25", [sumstats_train["QUANT_25"].mean()]*missing_data.height),
                                pl.Series("QUANT_75", [sumstats_train["QUANT_75"].mean()]*missing_data.height),
                                pl.Series("IDX_QUANT_0", [sumstats_train["IDX_QUANT_0"].mean()]*missing_data.height),
                                pl.Series("IDX_QUANT_100", [sumstats_train["IDX_QUANT_100"].mean()]*missing_data.height),
                                pl.Series("MIN_LOC", [sumstats_train["MIN_LOC"].mean()]*missing_data.height).cast(pl.Float64),
                                pl.Series("MAX_LOC", [sumstats_train["MAX_LOC"].mean()]*missing_data.height).cast(pl.Float64),
                                pl.Series("KURT", [sumstats_train["KURT"].mean()]*missing_data.height).cast(pl.Float64, strict=False),
                                pl.Series("SKEW", [sumstats_train["SKEW"].mean()]*missing_data.height).cast(pl.Float64, strict=False),
                                pl.Series("SD", [sumstats_train["SD"].mean()]*missing_data.height).cast(pl.Float64, strict=False),
                                pl.Series("MEAN_CHANGE", [sumstats_train["MEAN_CHANGE"].mean()]*missing_data.height).cast(pl.Float64, strict=False),
                                pl.Series("SUM_CHANGE", [0]*missing_data.height).cast(pl.Float64, strict=False),
                                pl.Series("LAST_VAL_DATE", [datetime(2013,1,1)]*missing_data.height).cast(pl.Date, strict=False),
                                pl.Series("SEQ_LEN", [0]*missing_data.height).cast(pl.Int64, strict=False),
                                pl.Series("FIRST_LAST", [0]*missing_data.height).cast(pl.Float64, strict=False),
                                pl.Series("ABNORM", [0]*missing_data.height).cast(pl.Int64, strict=False),
                            )
            )
            
            
        print(missing_data)
        sumstats = pl.concat([sumstats, missing_data.select(sumstats.columns)])
    print(sumstats)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    sumstats.write_parquet(args.res_dir+out_file_name+".parquet")