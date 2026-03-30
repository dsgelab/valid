 # Utils
import sys
from turtle import pd
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer, get_date, make_dir, init_logging, logging_print
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
# Reading etc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import warnings
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

def get_diag_data(file_path,
                  output_file_path,
                  diag_regex="",
                  chunk_size=10_000_000,
                  total_rows=0):
    count = 0
    writer = None
    timer = Timer()

    for chunk in pd.read_csv(file_path,
                            compression="gzip",
                            chunksize=chunk_size,
                            dtype={"CODE1": str, "CODE2": str, "SOURCE": str, "CATEGORY": str, "EVENT_AGE": float, "FID": str},
                            usecols=["FID","SOURCE","CODE1","CODE2","EVENT_AGE","CATEGORY","EVENT_DAY"]):
    
        bad_category = chunk["CATEGORY"].str.contains(r"^ICP|^OP|^MOP|None", regex=True)
        masks = []
        diag_sources = chunk["SOURCE"].isin(["INPAT", "OUTPAT", "PRIM_OUT", "REIMB"])
        code2_match = chunk["CODE2"].str.contains(diag_regex, regex=True, na=False)
        code1_match = chunk["CODE1"].str.contains(diag_regex, regex=True, na=False)
        masks.append((diag_sources & code2_match) | (diag_sources & code1_match) & ~bad_category)
        combined = masks[0]

        for crnt_mask in masks[1:]: combined |= crnt_mask

        chunk = chunk[combined].copy()
        chunk = chunk.melt(id_vars=["FID", "EVENT_AGE", "EVENT_DAY", "SOURCE"], 
                        value_vars=["CODE1", "CODE2"], 
                        value_name="CODE").drop(columns=["variable"])
        chunk = chunk[chunk["CODE"].str.contains(r"^[A-Z][0-9]", regex=True, na=False)].copy()
        chunk = chunk[chunk["CODE"].str.contains(diag_regex, regex=True, na=False)].copy()

        chunk = chunk.rename(columns={"FID": "FINNGENID"})
        chunk["EVENT_DAY"] = pd.to_datetime(chunk["EVENT_DAY"]).dt.date
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(output_file_path, table.schema)
        try:
            writer.write_table(table)
        except Exception as e:
            print(chunk)
            print(table)
            print(e)

        count += chunk_size
        if count % 100_000_000 == 0: print(f"Processed {count} rows. Time elapsed: {timer.get_elapsed()}")
        if total_rows > 0 and count >= total_rows: 
            logging_print(f"Finished writing N={total_rows}. Total time: {timer.get_elapsed()}")
            writer.close()
            return

    if writer:
        logging_print(f"Finished writing. Total time: {timer.get_elapsed()} and total rows: {count}")
        writer.close()


def get_med_data(file_path,
                  output_file_path,
                  med_regex="",
                  chunk_size=10_000_000,
                  total_rows=0):
    count = 0
    writer = None
    timer = Timer()

    for chunk in pd.read_csv(file_path,
                            compression="gzip",
                            chunksize=chunk_size,
                            dtype={"CODE1": str, "SOURCE": str, "EVENT_AGE": float, "FID": str},
                            usecols=["FID", "SOURCE", "CODE1",  "EVENT_AGE", "EVENT_DAY"]):
        
        mask = ((chunk["SOURCE"] == "PURCH") & chunk["CODE1"].str.contains(med_regex, regex=True, na=False))
        chunk = chunk[mask].copy()
        chunk = chunk[["FID", "EVENT_AGE", "EVENT_DAY", "SOURCE", "CODE1"]]
        chunk = chunk.rename(columns={"FID": "FINNGENID", "CODE1": "CODE"})
        chunk["EVENT_DAY"] = pd.to_datetime(chunk["EVENT_DAY"]).dt.date
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(output_file_path, table.schema)
        try:
            writer.write_table(table)
        except Exception as e:
            print(chunk)
            print(table)
            print(e)

        count += chunk_size
        if count % 100_000_000 == 0: print(f"Processed {count} rows. Time elapsed: {timer.get_elapsed()}")
        if total_rows > 0 and count >= total_rows: 
            logging_print(f"Finished writing N={total_rows}. Total time: {timer.get_elapsed()}")
            writer.close()
            return

    if writer:
        logging_print(f"Finished writing. Total time: {timer.get_elapsed()} and total rows: {count}")
        writer.close()


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Info for reading and naming
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Name of the laboratory", required=True)

    parser.add_argument("--icd_file_path", type=str, help="Path to the ICD file data.", required=True)
    parser.add_argument("--atc_file_path", type=str, help="Path to the ATC file data.", required=True)

    # Regex for selecting the data
    parser.add_argument("--diag_regex", type=str, help="Regex for selecting Code1 and 2 from the service sector detailed longitudinal data.", default="")
    parser.add_argument("--med_regex", type=str, help="Regex for selecting medication purchases.", required=False, default="")

    # Settings
    parser.add_argument("--chunk_size", type=int, help="Size of each chunk to process at a time.", required=False, default=10_000_000)
    parser.add_argument("--total_rows", type=int, help="Total number of rows to process. For testing purposes.", required=False, default=0)


    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    overall_timer = Timer()
    args = get_parser_arguments()
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.lab_name, logger, args)
    
    icd_out_file_path = args.res_dir + args.lab_name + "_diags_"+get_date()+".parquet"
    atc_out_file_path = args.res_dir + args.lab_name + "_meds_"+get_date()+".parquet"


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Getting & writing data                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.diag_regex != "":
        get_diag_data(args.icd_file_path, 
                      icd_out_file_path, 
                      diag_regex=args.diag_regex, 
                      chunk_size=args.chunk_size,
                      total_rows=args.total_rows)

    if args.med_regex !="":
        get_med_data(args.atc_file_path, 
                     atc_out_file_path, 
                     med_regex=args.med_regex, 
                     chunk_size=args.chunk_size, 
                     total_rows=args.total_rows)

    #Final logging
    logger.info("Time total: "+overall_timer.get_elapsed())
