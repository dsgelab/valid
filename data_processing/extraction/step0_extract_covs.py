# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/"))
from utils import *
# Standard stuff
import pandas as pd
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_covs_data(fg_ver="r12"):
    fg_ver = "r12"
    table = f"`finngen-production-library.sandbox_tools_{fg_ver}.minimum_extended_{fg_ver}_v1`"
    timer = Timer()
    query = f""" SELECT FINNGENID, APPROX_BIRTH_DATE, SEX, BMI, SMOKE2, COHORT, DEATH, AGE_AT_DEATH_OR_END_OF_FOLLOWUP
                    FROM {table}
                """
    data =  query_to_df(query)
    data = data.rename(columns={"APPROX_BIRTH_DATE": "DATE_OF_BIRTH", "SMOKE2":"SMOKE2", "AGE_AT_DEATH_OR_END_OF_FOLLOWUP": "END_OF_FOLLOWUP_AGE"})
    data.loc[:,"END_OF_FOLLOWUP"] = pd.to_datetime(data.DATE_OF_BIRTH) + data.END_OF_FOLLOWUP_AGE.apply(lambda x: relativedelta(days=x*365.25))
    data.drop(columns=["END_OF_FOLLOWUP_AGE"])
    return(data)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_ver", help="FinnGen release version", default="r12")
    parser.add_argument("--res_dir", type=str, help="Where data should be save", default="/home/ivm/PheRS/data/processed_data/step0/")

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
    file_name = "COVS_" + args.fg_ver + "_" + date
    log_file_name = "COVS_" + args.fg_ver + "_" +date_time
    make_dir(args.res_dir)
    make_dir(log_dir)
    
    init_logging(log_dir, log_file_name, date_time)
    data = get_covs_data(args.fg_ver)
    
    logger.info(print_count(data))
    data.to_csv(args.res_dir + file_name, sep=",", index=False)
# python3 /home/ivm/PheRS/scripts/extract/step0_extract_covs.py 
