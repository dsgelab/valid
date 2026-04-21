# Custom utils
import sys

sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, init_logging, Timer, read_file, print_count, logging_print

# Standard stuff
import polars as pl
import pandas as pd
import argparse
# Logging
import logging
logger = logging.getLogger(__name__)
from datetime import datetime

def read_start_date(start_date_str: str) -> datetime:
    """
    Function to read the start date from the input string.
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    return start_date

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--file_path_lab", type=str, help="Full path to the results lab measurement data.", required=True)
    parser.add_argument("--start_date", type=str, default="", help="Date to filter before", required=False)
    parser.add_argument("--egfr_data_path", type=str, default="/home/ivm/valid/data/processed_data/step1_clean/egfr_d1_KDIGO-soft_ld_2025-08-13.parquet", help="Path to processed eGFR data.", required=False)
    parser.add_argument("--select_omops_path", type=str, default="", help="Path to processed OMOP concept IDs data. To do batched work.", required=False)
    parser.add_argument("--fg_ver", type=str, default="R14", help="Version/SB. Can be R12-14 or ml4h.", required=False)

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
    init_logging(args.res_dir, "all", logger, args)

    out_file_name = "labs_"
    if args.start_date != "": 
        out_file_name = out_file_name + "pred" + str(int(args.start_date[0:4])+1) + "_"
    out_file_name += get_date()
    print(out_file_name)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    if args.select_omops_path == "":
        labs_data = read_file(args.file_path_lab)
        labs_data = labs_data.with_columns(pl.col.MEASUREMENT_VALUE.cast(pl.Float64),
                                          pl.col.OMOP_CONCEPT_ID.cast(pl.Utf8))
        if labs_data["APPROX_EVENT_DATETIME"].dtype == pl.Utf8:
            labs_data = labs_data.with_columns(pl.col.APPROX_EVENT_DATETIME.str.to_date("%Y-%m-%dT%H:%M", strict=False))

        # Adding manual eGFR data 
        crea_data = read_file(args.egfr_data_path)
        crea_data = (crea_data
                        .rename({"VALUE": "MEASUREMENT_VALUE", "DATE": "APPROX_EVENT_DATETIME"})
                        .with_columns(pl.Series("OMOP_CONCEPT_ID", ["40764999"]*crea_data.height),
                                    pl.col.APPROX_EVENT_DATETIME.dt.date().alias("APPROX_EVENT_DATETIME"))
                        )
        labs_data = (labs_data
                        .filter(~pl.col.OMOP_CONCEPT_ID.is_in(["40764999", "3020564", "46236952"]))
                        .with_columns(pl.col.APPROX_EVENT_DATETIME.dt.date().alias("APPROX_EVENT_DATETIME"))
                        )
        labs_data = pl.concat([labs_data.select("FINNGENID", "MEASUREMENT_VALUE", "APPROX_EVENT_DATETIME", "OMOP_CONCEPT_ID"),
                               crea_data.select("FINNGENID", "MEASUREMENT_VALUE", "APPROX_EVENT_DATETIME", "OMOP_CONCEPT_ID")])
    
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #                 Filter to before start of prediction                    #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #           
        labs_data = labs_data.filter(pl.col.APPROX_EVENT_DATETIME < read_start_date(args.start_date))
             
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #                 Stats for labs                                          #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
        labs_data = labs_data.group_by(["FINNGENID", "OMOP_CONCEPT_ID"]).agg(
                        pl.col.MEASUREMENT_VALUE.mean().alias("MEAN"),
                        pl.col.MEASUREMENT_VALUE.quantile(0.25).alias("QUANT25"),
                        pl.col.MEASUREMENT_VALUE.quantile(0.75).alias("QUANT75")
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
    else:
        select_omops = (read_file(args.select_omops_path)
                         .filter(pl.col.OMOP_ID>0)
                         # Removing NA units if there is also one without NA
                         .sort("OMOP_ID", "N").group_by("OMOP_ID", maintain_order=True).last()
                        )
        select_omops = dict(zip(select_omops["OMOP_ID"], select_omops["OMOP_UNIT"]))
        # remove OMOPs that have two entries, one of which is without unit
        labs_data = pl.DataFrame()
        total_timer = Timer()
        for crnt_omop, crnt_unit in select_omops.items():
            timer = Timer()
            # filter parquet to omop id, get mean, quantiles, pivot, rename, save with name including omop id
            if crnt_unit != "NA":
                crnt_labs_data = pl.DataFrame(pd.read_parquet(args.file_path_lab, 
                                                              filters=[("OMOP_CONCEPT_ID", "=", int(crnt_omop)),
                                                                       ("MEASUREMENT_UNIT_HARMONIZED", "=", crnt_unit)]))
            else:
                crnt_labs_data = pl.DataFrame(pd.read_parquet(args.file_path_lab, 
                                                              filters=[("OMOP_CONCEPT_ID", "=", int(crnt_omop))]))
                crnt_labs_data = crnt_labs_data.filter(pl.col("APPROX_EVENT_DATETIME").dt.date() < read_start_date(args.start_date))
            n_rows = crnt_labs_data.height
            crnt_labs_data = crnt_labs_data.group_by("FINNGENID").agg(
                pl.col("MEASUREMENT_VALUE").mean().alias("MEAN"),   
                pl.col("MEASUREMENT_VALUE").quantile(0.25).alias("QUANT25"),
                pl.col("MEASUREMENT_VALUE").quantile(0.75).alias("QUANT75")
            )
            crnt_labs_data.columns = ["FINNGENID", f"{crnt_omop}_MEAN", f"{crnt_omop}_QUANT25", f"{crnt_omop}_QUANT75"]
            if labs_data.height == 0:
                labs_data = crnt_labs_data
            else:
                labs_data = labs_data.join(crnt_labs_data, on="FINNGENID", how="full", coalesce=True)
            print(f"{crnt_omop} took {timer.get_elapsed()} have {n_rows} rows and N={crnt_labs_data.height} individuals")
        logging_print(f"Total time for looping through omops: {total_timer.get_elapsed()}")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    labs_data.write_parquet(args.res_dir+out_file_name+".parquet")