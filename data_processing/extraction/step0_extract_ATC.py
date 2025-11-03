# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import *
# Standard stuff
import pandas as pd
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
from datetime import datetime

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_ver", help="FinnGen release version", default="r12")
    parser.add_argument("--res_dir", type=str, help="Where data should be save", default="/home/ivm/valid/data/extra_data/processed_data/step0_extract/")

    args = parser.parse_args()

    return(args)

def init_logging(log_dir, log_file_name, date_time):
    logging.basicConfig(filename=log_dir+log_file_name+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.info("Time: " + date_time + " Args: --" + ' --'.join(f'{k}={v}' for k, v in vars(args).items()))
     
if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    
    log_dir = args.res_dir + "logs/"
    date = datetime.today().strftime("%Y-%m-%d")
    date_time = datetime.today().strftime("%Y-%m-%d-%H%M")
    file_name = "ATC_long_" + args.fg_ver + "_" + date
    log_file_name = "ATC_long_" + args.fg_ver + "_" +date_time
    make_dir(args.res_dir)
    make_dir(log_dir)
    
    init_logging(log_dir, log_file_name, date_time)
    
    table = f"`finngen-production-library.sandbox_tools_{args.fg_ver}.finngen_{args.fg_ver}_service_sector_detailed_longitudinal_v1`"
    query_pat = f"""SELECT FINNGENID, EVENT_AGE, APPROX_EVENT_DAY, CODE1
                     FROM {table}
                     WHERE SOURCE = 'PURCH' AND CODE1 != 'NA'
                 """
    all_atcs = query_to_df(query_pat)
    logger.info("Time import: "+timer.get_elapsed())
    print(all_atcs)
    all_atcs=all_atcs.rename({"CODE1": "ATC_CODE", "APPROX_EVENT_DAY": "DATE"}).unique()
    print(all_atcs)
    all_atcs.write_parquet(args.res_dir + file_name + ".parquet")
    init_logging(f"Number of rows {all_atcs.height} with {all_atcs["FINNGENID"].unique().len()} individuals.")

# python3 /home/ivm/valid/data/extra_data/scripts/extract/step0_extract_ATC.py --fg_ver r13