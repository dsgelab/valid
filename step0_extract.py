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

"""Get data for a specific OMOP concept ID from the kanta lab data. Returns a panda with data."""
def get_omop_id_data(omop_concept_ids = ["3020564"],
                     table = f"`finngen-production-library.sandbox_tools_r12.kanta_r12_v1`",
                     columns = ["FINNGENID", "SEX", "EVENT_AGE", "TEST_OUTCOME_IMPUTED", "MEASUREMENT_VALUE_HARMONIZED",  "MEASUREMENT_UNIT_HARMONIZED", "MEASUREMENT_VALUE", "MEASUREMENT_UNIT", "APPROX_EVENT_DATETIME"]):
    omop_concept_str = ",".join(f"'{id}'" for id in omop_concept_ids)
    col_str = ", ".join(columns)
    query = f""" SELECT {col_str}
                 FROM {table}
                 WHERE CAST({table}.OMOP_CONCEPT_ID as STRING) IN ({omop_concept_str})
                 ORDER BY FINNGENID, APPROX_EVENT_DATETIME
         	 """
    timer = Timer()
    data = query_to_df(query)
    logger.info("Time import: "+timer.get_elapsed())
    return(data)

"""Setting up the parser arguments."""
def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--omop", help="OMOP ID to extract", required=True)
    parser.add_argument("--res_dir", help="Path to the results directory.", required=True)
    parser.add_argument("--lab_name", help="Readable name of the measurement value.", required=True)
    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    #### Setup
    timer = Timer()
    args = get_parser_arguments()
    ## Paths
    log_dir = args.res_dir + "logs/"
    file_name = args.lab_name + "_" + get_date()
    log_file_name = args.lab_name + "_" + get_datetime()
    make_dir(log_dir)
    make_dir(args.res_dir)
    ## Logging
    init_logging(log_dir, log_file_name, logger)
    
    #### Data processing
    ## Raw data
    data = get_omop_id_data(omop_concept_ids=[args.omop], columns=["FINNGENID", "SEX", "EVENT_AGE", "TEST_OUTCOME_IMPUTED", "MEASUREMENT_VALUE_HARMONIZED", "MEASUREMENT_UNIT_HARMONIZED", "APPROX_EVENT_DATETIME"])
    ## Sorting
    data = data.sort_values(["FINNGENID", "APPROX_EVENT_DATETIME"], ascending=False)
    data = data.rename(columns={"APPROX_EVENT_DATETIME": "DATE", "MEASUREMENT_VALUE_HARMONIZED": "VALUE", "MEASUREMENT_UNIT_HARMONIZED": "UNIT", "TEST_OUTCOME_IMPUTED":"ABNORM"})
    data = data[["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "UNIT", "ABNORM"]]

    ## Saving
    data.to_csv(args.res_dir + file_name + ".csv", sep=",", index=False)
    logger.info("Time total: "+timer.get_elapsed())