# Custom utils
import sys

sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, init_logging, Timer, read_file, print_count

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
    parser.add_argument("--clean", type=int, default=0, help="How many predictors to remove.", required=False)
    parser.add_argument("--mean_impute", type=int, default=0, help="Impute mean for each OMOP if missing for individual.", required=False)
    parser.add_argument("--egfr_data_path", type=str, default="/home/ivm/valid/data/processed_data/step1_clean/egfr_d1_KDIGO-soft_ld_2025-08-13.parquet", help="Path to processed eGFR data.", required=False)

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
    out_file_name = args.file_name_labels_start+"_labs_"
    if args.mean_impute == 1:
        out_file_name += "impute_"
    if args.clean == 1:
        out_file_name += "clean_"
    out_file_name += get_date()
    print(out_file_name)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    labs_data = read_file(args.file_path_lab)
    labs_data = labs_data.with_columns(pl.col.MEASUREMENT_VALUE_HARMONIZED.cast(pl.Float64),
                                       pl.col.OMOP_CONCEPT_ID.cast(pl.Utf8))

    # Adding manual eGFR data 
    if args.lab_name not in ["krea", "egfr"]:
        crea_data = read_file(args.egfr_data_path)
        crea_data = (crea_data
                     .rename({"VALUE": "MEASUREMENT_VALUE_HARMONIZED", "DATE": "APPROX_EVENT_DATETIME"})
                     .with_columns(pl.Series("OMOP_CONCEPT_ID", ["40764999"]*crea_data.height),
                                   pl.col.APPROX_EVENT_DATETIME.dt.date().alias("APPROX_EVENT_DATETIME"))
                    )
        labs_data = (labs_data
                     .filter(~pl.col.OMOP_CONCEPT_ID.is_in(["40764999", "3020564", "46236952"]))
                     .with_columns(pl.col.APPROX_EVENT_DATETIME.dt.date().alias("APPROX_EVENT_DATETIME"))
                    )
        labs_data = pl.concat([labs_data.select("FINNGENID", "MEASUREMENT_VALUE_HARMONIZED", "APPROX_EVENT_DATETIME", "OMOP_CONCEPT_ID"),
                                crea_data.select("FINNGENID", "MEASUREMENT_VALUE_HARMONIZED", "APPROX_EVENT_DATETIME", "OMOP_CONCEPT_ID")])
        


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Removing duplicate predictors                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.lab_name == "krea" or args.lab_name == "egfr":
        if args.clean == 1:
            # krea, egfr some other formula, egfr, cystatin c, UACR
            labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["40764999", "46236952", "3020564", "3030366", "3020682"]))
        else:
            labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["40764999", "46236952", "3020564"]))
    if args.lab_name == "hba1c":
        if args.clean == 1:
            labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3004410", "3018251", "3013826", "3013826"])) # hba1c and fasting glucose, glucose, glucose 2 hours post dose
        else:
            labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3004410", "3018251"])) # hba1c and fasting glucose
    if args.lab_name == "alatasat":
        # ALAT, ASATs
        labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3006923", "3013721"]))
    if args.lab_name == "tsh":
        if args.clean == 1:
            # tsh, t4, t3
            labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3009201", "3008486", "3026989"]))
        else:
            labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3009201", "3008486"]))

    if args.lab_name == "ldl":
        if args.clean == 1: 
            # LDL, LDL/HDL, HDl standard, HDL total, cholesterol total, weird cholesterol non hdl rare, tri fast, tri
            labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3001308", "3019900", "3023602", "42868674","3019900", "3048773", "3025839"]))
        else:
            # LDL, tryg free, tryg
            labs_data = labs_data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3001308", "3048773", "3025839"]))
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Filter to before start of prediction                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    labels = read_file(args.dir_path_labels+args.file_name_labels_start+"_labels.parquet")
    if args.start_date == "": 
        labs_data = labs_data.join(labels.select("FINNGENID"), on="FINNGENID", how="right")
        labs_data = labs_data.filter(pl.col.APPROX_EVENT_DATETIME < pl.col.START_DATE)
    else:
        try:
            labs_data = labs_data.filter(pl.col.APPROX_EVENT_DATETIME < datetime.strptime(args.start_date, "%Y-%m-%d"))
        except:
            labs_data = labs_data.filter(pl.col.APPROX_EVENT_DATETIME < datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S"))


            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Stats for labs                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    labs_data = labs_data.group_by(["FINNGENID", "OMOP_CONCEPT_ID"]).agg(
                    pl.col.MEASUREMENT_VALUE_HARMONIZED.mean().alias("MEAN"),
                    pl.col.MEASUREMENT_VALUE_HARMONIZED.quantile(0.25).alias("QUANT25"),
                    pl.col.MEASUREMENT_VALUE_HARMONIZED.quantile(0.75).alias("QUANT75")
    )
    print_count(labs_data)

    # Pivot the data wide
    labs_data = (labs_data
                 .pivot(values=["MEAN", "QUANT25", "QUANT75"], index="FINNGENID", on="OMOP_CONCEPT_ID")
    )
    # Rename columns to include the statistic in the name
    labs_data = labs_data.rename({
        f"{col_name}": f"{col_name.split("_")[1]}_{col_name.split("_")[0]}"
        for col_name in labs_data.columns if col_name != "FINNGENID"
    })

    if args.mean_impute == 1:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #                 Adding individuals with no measurements                 #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
        labs_data = labs_data.join(labels.select("FINNGENID", "SET"), on="FINNGENID", how="right")
    
        print_count(labs_data)
        # Mean impute with help of co-pilot 16.9.25
        col_means = {
            col: labs_data.filter(pl.col.SET==0)[col].mean()
            for col in labs_data.columns if col != "FINNGENID"
        }
        # Fill missing values with column means
        labs_data = labs_data.with_columns([
            pl.col(col).fill_null(col_means[col]).alias(col) for col in labs_data.columns if col != "FINNGENID"
        ]).drop("SET")
    print(labs_data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    labs_data.write_parquet(args.res_dir+out_file_name+".parquet")