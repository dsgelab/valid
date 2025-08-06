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
    parser.add_argument("--file_path_lab", type=str, help="Full üath to the results lab measurement data.", required=True)
    parser.add_argument("--dir_path_labels", type=str, help="Path to directory with label data.", required=True)
    parser.add_argument("--file_name_labels_start", type=str, help="Path to the diagnosis data file", default="")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
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
    init_logging(args.res_dir, args.lab_name, logger, args)
    out_file_name = args.file_name_labels_start+"_labs_"+get_date()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    labs_data = read_file(args.file_path_lab)
    labs_data = labs_data.with_columns(pl.col.MEASUREMENT_VALUE_HARMONIZED.cast(pl.Float64))
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Filter to before start of prediction                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    labels = read_file(args.dir_path_labels+args.file_name_labels_start+"_labels.parquet")
    if args.start_date == "": 
        labs_data = labs_data.join(labels.select("FINNGENID", "START_DATE"), on="FINNGENID", how="left")
        labs_data = labs_data.filter(pl.col.APPROX_EVENT_DATETIME < pl.col.START_DATE)
    else:
        labs_data = labs_data.filter(pl.col.APPROX_EVENT_DATETIME < datetime.strptime(args.start_date, "%Y-%m-%d"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Removing duplicate predictors                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.lab_name == "krea" or args.lab_name == "egfr":
        # krea, egfr, cystatin c
        labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["40764999", "3020564", "3030366"]))
    if args.lab_name == "hba1c":
        labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3004410", "3018251"])) # hba1c and fasting glucose
    if args.lab_name == "alatasat":
        # ALAT, ASATs
        labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3006923", "3013721"]))
    if args.lab_name == "tsh":
        labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3009201", "3008486"]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Stats for labs                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    labs_data = labs_data.group_by(["FINNGENID", "OMOP_CONCEPT_ID"]).agg(
                    pl.col.MEASUREMENT_VALUE_HARMONIZED.mean().alias("MEAN"),
                    pl.col.MEASUREMENT_VALUE_HARMONIZED.quantile(0.25).alias("QUANT25"),
                    pl.col.MEASUREMENT_VALUE_HARMONIZED.quantile(0.75).alias("QUANT75")
    )
    # Pivot the data wide
    labs_data = (labs_data
                 .pivot(values=["MEAN", "QUANT25", "QUANT75"], index="FINNGENID", on="OMOP_CONCEPT_ID")
    )
    # Rename columns to include the statistic in the name
    labs_data = labs_data.rename({
        f"{col_name}": f"{col_name.split("_")[1]}_{col_name.split("_")[0]}"
        for col_name in labs_data.columns if col_name != "FINNGENID"
    })
    print(labs_data)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    labs_data.write_parquet(args.res_dir+out_file_name+".parquet")