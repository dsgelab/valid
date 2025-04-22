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
    parser.add_argument("--file_path_lab", type=str, help="Full üath to the results lab measurement data.", required=True)
    parser.add_argument("--dir_path_labels", type=str, help="Path to directory with label data.", required=True)
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
    out_file_name = args.file_name_start+"_labs_"+get_date()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    labs_data = read_file(args.file_path_lab)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Filter to before start of prediction                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    labels = read_file(args.file_path+args.file_name_start+"_labels.parquet")
    labs_data = labs_data.join(labels, on="FINNGENID", how="right", coalesce=True)
    labs_data = labs_data.filter(pl.col.APPROX_EVENT_DATETIME < pl.col.START_DATE)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Removing duplicate predictors                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.lab_name == "krea" or args.lab_name == "egfr":
        # krea, egfr
        labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["40764999", "3020564"]))
    if args.lab_name == "hba1c":
        labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3004410"]))
    if args.lab_name == "alatasat":
        # ALAT, ASATs
        labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3006923", "3013721"]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Stats for labs                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    labs_data = labs_data.group_by(["FINNGENID", "OMOP_CONCEPT_ID"]).agg(
                    pl.col.MEASUREMENT_VALUE_HARMONIZED.mean().alias("MEAN").over("FINNGENID"),
                    pl.col.MEASUREMENT_VALUE_HARMONIZED.quantile(0.25).alias("QUANT_25"),
                    pl.col.MEASUREMENT_VALUE_HARMONIZED.quantile(0.75).alias("QUANT_75")
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    labs_data.write_parquet(args.res_dir+out_file_name+".parquet")